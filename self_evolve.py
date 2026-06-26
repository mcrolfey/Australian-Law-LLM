"""
Australian Law LLM - Recursive Self-Evolution Loop
====================================================
Trains the model for N steps, sends the loss curve and current config to a
local LM Studio model, which suggests hyperparameter improvements, then
retrains with the updated config. Repeats for K rounds.

Prerequisites:
  1. LM Studio running with a model loaded (default: http://localhost:1234)
  2. Kaggle corpus cached locally from a previous run
  3. pip install openai

Usage:
    python self_evolve.py --gpu 8gb --rounds 5 --steps-per-round 200
    python self_evolve.py --gpu 8gb --rounds 3 --steps-per-round 100 --lm-studio-url http://localhost:1234
    python self_evolve.py --list-configs
"""

import argparse
import copy
import gc
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Windows import order fix (triton before torch before unsloth)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import triton
    from torch._inductor.runtime.hints import DeviceProperties
except Exception:
    pass

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Config bounds — LM Studio cannot push values outside these ranges
# ──────────────────────────────────────────────────────────────────────────────
BOUNDS = {
    "learning_rate":                  (1e-6, 1e-3),
    "per_device_train_batch_size":    (1, 4),
    "gradient_accumulation_steps":    (1, 32),
    "warmup_steps":                   (0, 200),
    "weight_decay":                   (0.0, 0.3),
    "r":                              (4, 64),
    "lora_alpha":                     (4, 128),
}

TUNABLE_KEYS = set(BOUNDS.keys())

GPU_CONFIGS = {
    "4gb":  "configs.gpu_4gb",
    "8gb":  "configs.gpu_8gb",
    "16gb": "configs.gpu_16gb",
    "24gb": "configs.gpu_24gb",
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Recursive self-evolution training loop")
    p.add_argument("--gpu", choices=list(GPU_CONFIGS.keys()), default="8gb")
    p.add_argument("--rounds", type=int, default=5,
                   help="Number of evolution rounds (default: 5)")
    p.add_argument("--steps-per-round", type=int, default=200,
                   help="Training steps per round (default: 200)")
    p.add_argument("--lm-studio-url", default="http://localhost:1234",
                   help="LM Studio base URL (default: http://localhost:1234)")
    p.add_argument("--lm-studio-model", default=None,
                   help="Model ID in LM Studio (auto-detected if omitted)")
    p.add_argument("--output-dir", default="self_evolve_output",
                   help="Directory for adapter checkpoints and logs (default: self_evolve_output)")
    p.add_argument("--list-configs", action="store_true",
                   help="Print available GPU configs and exit")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# LM Studio client
# ──────────────────────────────────────────────────────────────────────────────
def get_lm_studio_client(base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)
    return OpenAI(base_url=f"{base_url}/v1", api_key="lm-studio")


def detect_lm_studio_model(client) -> str:
    try:
        models = client.models.list()
        if not models.data:
            print("ERROR: No model loaded in LM Studio. Load a model and try again.")
            sys.exit(1)
        model_id = models.data[0].id
        print(f"  LM Studio model detected: {model_id}")
        return model_id
    except Exception as e:
        print(f"ERROR: Cannot connect to LM Studio: {e}")
        print("Make sure LM Studio is running with a model loaded.")
        sys.exit(1)


def ask_lm_studio(client, model_id: str, prompt: str) -> str:
    # Try full prompt first; if LM Studio rejects due to context length,
    # fall back to progressively shorter answer truncations.
    for char_limit in [None, 500, 200, 100]:
        if char_limit is not None:
            # Truncate each answer block to char_limit characters
            truncated = []
            for line in prompt.split("\n"):
                if line.startswith("A: ") and len(line) > char_limit:
                    truncated.append(line[:char_limit] + "…")
                else:
                    truncated.append(line)
            send_prompt = "\n".join(truncated)
            print(f"  [Retrying with answers truncated to {char_limit} chars]")
        else:
            send_prompt = prompt
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": send_prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "n_ctx" in str(e) or "context length" in str(e).lower() or "400" in str(e):
                if char_limit == 100:
                    print("  ERROR: prompt still too long even after truncation.")
                    print("  Fix: in LM Studio, reload the model with a larger context (e.g. 32768).")
                    raise
                continue
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Config management
# ──────────────────────────────────────────────────────────────────────────────
def load_base_config(gpu_tier: str) -> dict:
    module = importlib.import_module(GPU_CONFIGS[gpu_tier])
    return copy.deepcopy(module.CONFIG)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def apply_patch(cfg: dict, patch: dict) -> tuple[dict, list[str]]:
    """Apply a config patch from LM Studio, clamping values to safe bounds."""
    new_cfg = copy.deepcopy(cfg)
    applied = []
    for key, value in patch.items():
        if key not in TUNABLE_KEYS:
            continue
        lo, hi = BOUNDS[key]
        if isinstance(value, float) or isinstance(value, int):
            clamped = clamp(value, lo, hi)
            if clamped != new_cfg.get(key):
                applied.append(f"  {key}: {new_cfg.get(key)} → {clamped}")
                new_cfg[key] = type(new_cfg.get(key, value))(clamped)
    return new_cfg, applied


def build_lm_studio_prompt(round_num: int, cfg: dict, loss_log: list[dict],
                            history: list[dict], sample_answers: str = "") -> str:
    current_config_str = json.dumps(
        {k: cfg[k] for k in TUNABLE_KEYS if k in cfg}, indent=2
    )
    loss_summary = "\n".join(
        f"  step {e['step']:4d}: loss={e['loss']:.4f}" for e in loss_log
    )
    if history:
        history_str = ""
        for h in history[-3:]:  # show last 3 rounds
            history_str += (
                f"\n  Round {h['round']}: changes={h['changes']}, "
                f"start_loss={h['start_loss']:.4f}, end_loss={h['end_loss']:.4f}"
            )
    else:
        history_str = "  (no previous rounds)"

    sample_section = f"\n20 representative benchmark answers (every 5th question, answers capped at 600 chars):\n{sample_answers}" if sample_answers else ""

    return f"""You are an expert machine learning engineer helping to optimise hyperparameters for a LLaMA LoRA fine-tuning run on Australian legal text using Unsloth on a Windows machine with an RTX 3070 (8GB VRAM).

TRAINING SUMMARY — Round {round_num}
======================================
Current config:
{current_config_str}

Loss curve this round:
{loss_summary}

Previous rounds:
{history_str}
{sample_section}

TASK
====
Analyse the loss curve and the sample answers above. Consider both whether the loss is decreasing AND whether the answers look factually grounded and coherent. If answers are repetitive or hallucinating, suggest a higher repetition penalty or lower learning rate. If the loss is still decreasing steeply, suggest continuing with similar settings.

Respond with ONLY a JSON object containing the keys you want to change from the current config. Only include keys that need changing. Use these exact key names:
- learning_rate (float, range 1e-6 to 1e-3)
- per_device_train_batch_size (int, 1-4)
- gradient_accumulation_steps (int, 1-32)
- warmup_steps (int, 0-200)
- weight_decay (float, 0.0-0.3)
- r (int, LoRA rank, 4-64)
- lora_alpha (int, 4-128)

If no changes are needed, return: {{}}

Respond with ONLY the JSON object, no explanation, no markdown fences.
"""


def extract_json(text: str) -> dict:
    """Extract a JSON object from LM Studio's response."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — loaded once, reused across all rounds
# ──────────────────────────────────────────────────────────────────────────────
ALPACA_PROMPT = """\
Below is an instruction that describes a legal task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the following Australian legal document. Synthesize its legal principles based on the provided jurisdiction and context framework.

### Input:
Citation: {}
Jurisdiction: {}
Document Type: {}

### Response:
{}"""


KAGGLE_CACHE = os.path.expanduser(
    "~/.cache/kagglehub/datasets/umarbutler/open-australian-legal-corpus/versions/2/corpus.jsonl"
)


def load_and_map_dataset(tokenizer, trunc_len: int):
    """Download (or use cache) and map the corpus. Called once before the loop."""
    from datasets import load_dataset

    print("Loading Open Australian Legal Corpus (once — reused every round)...")

    # Use local cache if available — avoids Kaggle DNS failures when offline
    if os.path.exists(KAGGLE_CACHE):
        corpus_path = KAGGLE_CACHE
        print(f"  Using cached corpus: {corpus_path}")
    else:
        import kagglehub
        dataset_dir = kagglehub.dataset_download("umarbutler/open-australian-legal-corpus")
        corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    dataset = load_dataset("json", data_files=corpus_path, split="train")

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        citations     = examples.get("citation",     ["Unknown"] * len(examples["text"]))
        jurisdictions = examples.get("jurisdiction", ["Unknown"] * len(examples["text"]))
        types         = examples.get("type",         ["Unknown"] * len(examples["text"]))
        texts         = examples["text"]
        formatted = []
        for citation, jurisdiction, doc_type, text in zip(citations, jurisdictions, types, texts):
            raw = ALPACA_PROMPT.format(citation, jurisdiction, doc_type, text or "") + EOS_TOKEN
            tokens = tokenizer.encode(raw, truncation=True, max_length=trunc_len)
            formatted.append(tokenizer.decode(tokens, skip_special_tokens=False))
        return {"text": formatted}

    print("Mapping dataset (this runs once)...")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"  Dataset ready: {len(dataset)} documents\n")
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark questions (same 100 as batch_test.py)
# ──────────────────────────────────────────────────────────────────────────────
QUESTIONS = [
    "Does the Australian Constitution contain a comprehensive Bill of Rights?",
    "What is the 'implied freedom of political communication' in Australian constitutional law?",
    "Explain the constitutional significance of Section 51(xxix) (the external affairs power).",
    "Under Section 109 of the Australian Constitution, what happens when a State law is inconsistent with a Commonwealth law?",
    "How is the separation of powers applied differently in Australia compared to the United States?",
    "What was the legal and historical significance of the Tasmanian Dam Case (1983)?",
    "Explain the 'Engineers Case' (1920) and how it changed constitutional interpretation in Australia.",
    "What is the role of the Governor-General under Section 61 of the Constitution?",
    "What specific rights does Section 92 of the Constitution guarantee regarding trade and commerce?",
    "Can the Australian Constitution be amended by an Act of Parliament? Explain the Section 128 mechanism.",
    "What is the difference between an 'indictable offence' and a 'summary offence' in Australia?",
    "Is the criminal code uniform across all Australian states and territories? Explain the divide.",
    "How does the defence of provocation operate in New South Wales compared to Victoria?",
    "Define the concept of doli incapax as applied in Australian jurisdictions.",
    "What are the specific elements of murder under the Crimes Act 1900 (NSW)?",
    "What is the role of the Director of Public Prosecutions (DPP) in Australia?",
    "Explain the 'Golden Thread' principle in Australian criminal justice (referencing Woolmington).",
    "How is the concept of 'joint criminal enterprise' defined under Australian common law?",
    "What is the objective test for criminal negligence in Australian law?",
    "Does Australia have a universal right to silence when questioned by police? How do 'special warning' provisions work in NSW?",
    "What constitutes 'unconscionable conduct' under the Australian Consumer Law (ACL)?",
    "How are consumer guarantees applied under Schedule 2 of the Competition and Consumer Act 2010?",
    "What is the legal definition of a 'consumer' under the updated Australian Consumer Law?",
    "Describe the landmark equitable rule established in Commercial Bank of Australia Ltd v Amadio (1983).",
    "What is the doctrine of promissory estoppel in Australian law following Waltons Stores (Interstate) Ltd v Maher?",
    "What are the available remedies for a 'major failure' of a consumer guarantee under the ACL?",
    "Explain the doctrine of privity of contract as it currently stands in Australian law.",
    "What is the difference between a condition and a warranty in Australian contract law?",
    "How does the Australian Competition and Consumer Commission (ACCC) enforce anti-competitive behavior?",
    "Does the postal acceptance rule still apply to email communications under the Electronic Transactions Act 1999 (Cth)?",
    "How did the Civil Liability Act 2002 (NSW) alter the common law test for the duty of care?",
    "Explain how a 'duty of care' is established in Australia following the High Court's decision in Sullivan v Moody.",
    "What are the specific defences available under the uniform Defamation Act 2005 across Australian states?",
    "How does the defence of 'honest opinion' work in Australian defamation law?",
    "What is the test for factual causation in Australian tort law (the 'necessary condition' test)?",
    "Explain the concept of 'vicarious liability' in an Australian employment context.",
    "How are damages for non-economic loss capped or calculated under Australian personal injury legislation?",
    "What is the 'volenti non fit injuria' (voluntary assumption of risk) defence under the Civil Liability Acts?",
    "How is the standard of care for professionals determined in Australia (the modified Bolam test / Rogers v Whitaker)?",
    "What are the elements of the tort of deceit in Australian common law?",
    "What are the primary statutory duties of a company director under sections 180-183 of the Corporations Act 2001 (Cth)?",
    "Explain the 'business judgment rule' under Section 180(2) of the Corporations Act.",
    "What is the exact role and jurisdiction of the Australian Securities and Investments Commission (ASIC)?",
    "Describe the legal structure and limitations of a 'proprietary limited' (Pty Ltd) company in Australia.",
    "Under what specific circumstances can the corporate veil be pierced under Australian law?",
    "What constitutes insolvent trading, and what are the director's liabilities under Section 588G of the Corporations Act?",
    "How does the process of Voluntary Administration work in Australia, and how does it differ from US Chapter 11?",
    "What are the requirements for 'continuous disclosure' under the ASX Listing Rules?",
    "How does the Personal Property Securities Act 2009 (Cth) (PPSA) operate regarding security interests?",
    "What is a 'scheme of arrangement' under Part 5.1 of the Corporations Act?",
    "Explain the 'Torrens title system' used in Australian real estate.",
    "What does the concept of 'indefeasibility of title' mean in Australian property law?",
    "What was the legal and historical significance of the High Court's decision in Mabo v Queensland (No 2) (1992)?",
    "How is 'native title' defined and claimed under the Native Title Act 1993 (Cth)?",
    "What is the legal difference between a joint tenancy and a tenancy in common in Australia?",
    "Explain the doctrine of adverse possession as it applies in Victoria or New South Wales.",
    "What is a 'caveat', and how is it used to protect unregistered interests in the Australian property system?",
    "How are easements legally created under Australian property law?",
    "What rights and statutory obligations does a mortgagee have when exercising a power of sale in Australia?",
    "Does the rule against perpetuities still exist in Australian states, and if so, how has it been modified?",
    "What does the 'no-fault' divorce principle mean under the Family Law Act 1975 (Cth)?",
    "How do Australian courts determine the 'best interests of the child' under Section 60CC?",
    "What factors must the court consider in a property settlement under Section 79 of the Family Law Act?",
    "How is a 'de facto relationship' legally defined under Australian federal law?",
    "What is the role of an Independent Children's Lawyer (ICL) in Australian family court proceedings?",
    "How are superannuation interests treated in Australian family law property settlements?",
    "Explain the presumption of 'equal shared parental responsibility' (and note any recent legislative changes to this concept).",
    "What constitutes 'family violence' under the Family Law Act 1975?",
    "What are the strict legal requirements to make a Binding Financial Agreement (BFA) enforceable in Australia?",
    "What is the jurisdictional role of the Federal Circuit and Family Court of Australia (FCFCOA)?",
    "What is the fundamental difference between 'merits review' and 'judicial review' in Australia?",
    "Explain the jurisdiction and function of the Administrative Appeals Tribunal (AAT) (or the new Administrative Review Tribunal).",
    "What are the specific grounds for judicial review under the Administrative Decisions (Judicial Review) Act 1977 (ADJR Act)?",
    "How does the concept of 'procedural fairness' or 'natural justice' apply to Australian administrative decision-makers?",
    "What is the 'hearing rule' and the 'bias rule' in Australian administrative law?",
    "What was the legal significance of Minister for Immigration and Citizenship v Li (2013) regarding legal unreasonableness?",
    "Explain the concept of 'jurisdictional error' in Australian law.",
    "How do 'privative clauses' operate, and how did the High Court treat them in Plaintiff S157/2002?",
    "What is the role of the Commonwealth Ombudsman?",
    "How is legal standing (locus standi) established for judicial review in Australian courts?",
    "Does Australia require copyright registration for copyright protection to apply to a work?",
    "What constitutes 'fair dealing' under the Copyright Act 1968 (Cth), and how does it differ from US 'fair use'?",
    "How long does a standard patent last under the Patents Act 1990 (Cth)?",
    "What are the Australian Privacy Principles (APPs) under the Privacy Act 1988?",
    "Explain the concept of 'moral rights' as protected under Australian copyright law.",
    "What is the primary role of the Fair Work Commission in Australia?",
    "List and explain three of the National Employment Standards (NES) under the Fair Work Act 2009.",
    "What legally constitutes an 'unfair dismissal' in Australia?",
    "How does the common law tort of 'passing off' operate in Australia compared to statutory trademark infringement?",
    "Are non-compete clauses automatically enforceable in Australian employment contracts? Explain the restraint of trade doctrine.",
    "Format a standard AGLC4 citation for a High Court of Australia case decided in 2022.",
    "Explain the hierarchy of courts in the state of Victoria, from the lowest court to the highest.",
    "Draft a brief boilerplate jurisdiction and governing law clause for a commercial contract specifying New South Wales.",
    "What is the purpose of a 'Statement of Claim' versus a 'Notice of Motion' in Australian civil procedure?",
    "How does the practice of 'discovery' operate under the Federal Court Rules 2011?",
    "What is the 'Uniform Evidence Law' framework, and which Australian jurisdictions have adopted it?",
    "How is legal professional privilege defined and applied under the Evidence Act 1995 (Cth)?",
    "What are the formal execution requirements for a valid deed in Queensland?",
    "Explain the strict professional distinction between a 'barrister' and a 'solicitor' in the Australian legal system.",
    "What is a 'costs order' in Australian civil litigation, and what does 'costs follow the event' mean?",
]

EVAL_PROMPT_TEMPLATE = """\
Below is an instruction that describes a legal task. Write a response that appropriately completes the request.

### Instruction:
You are an expert Australian legal assistant trained on the Open Australian Legal Corpus. \
Answer the following question accurately, citing relevant legislation or case law where appropriate.

### Input:
{question}

### Response:
"""


# ──────────────────────────────────────────────────────────────────────────────
# Post-round evaluation — run all 100 questions
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_round(cfg: dict, adapter_dir: str, output_dir: Path,
                   round_num: int) -> list[dict]:
    """Load the just-trained adapter and run all 100 benchmark questions."""
    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"\n  Running benchmark evaluation ({len(QUESTIONS)} questions)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=cfg["dtype"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    results = []
    t0 = time.time()
    for i, question in enumerate(QUESTIONS, start=1):
        prompt = EVAL_PROMPT_TEMPLATE.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=0.2,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        results.append({"q": i, "question": question, "answer": answer})
        if i % 10 == 0:
            print(f"    {i}/{len(QUESTIONS)}  ({(time.time()-t0)/60:.1f} min elapsed)")

    # Save readable txt
    txt_path = output_dir / f"round_{round_num:02d}_answers.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"ROUND {round_num} — BENCHMARK ANSWERS\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"Q{r['q']:03d}. {r['question']}\n")
            f.write("-" * 70 + "\n")
            f.write(r["answer"] + "\n\n")

    # Save JSON
    json_path = output_dir / f"round_{round_num:02d}_answers.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Evaluation done in {(time.time()-t0)/60:.1f} min")
    print(f"  Results saved → {txt_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def run_training_round(cfg: dict, steps: int, adapter_dir: str, dataset,
                       prev_adapter_dir: str = None) -> list[dict]:
    """
    Run one training round with a pre-mapped dataset.
    If prev_adapter_dir is set, merges the previous round's adapter into the
    base model before attaching fresh LoRA — so each round continues from
    where the last one left off rather than restarting from scratch.
    Returns the loss log as a list of {"step": int, "loss": float} dicts.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    gc.collect()
    torch.cuda.empty_cache()

    # Load model
    print(f"\n  Loading {cfg['model_name']} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=cfg["dtype"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    tokenizer.padding_side = "right"

    # Always attach fresh LoRA via get_peft_model so Unsloth's kernels are applied.
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 3407),
    )

    if prev_adapter_dir and os.path.isdir(prev_adapter_dir):
        # Copy previous round's adapter weights into the fresh LoRA.
        # This gives cumulative learning without touching the base model weights
        # or creating a nested PeftModel — Unsloth kernels remain intact.
        print(f"  Seeding LoRA weights from: {prev_adapter_dir}")
        import glob as _glob
        from peft import set_peft_model_state_dict
        weight_files = _glob.glob(os.path.join(prev_adapter_dir, "*.safetensors"))
        if weight_files:
            from safetensors.torch import load_file as _load_safetensors
            prev_weights = {}
            for wf in weight_files:
                prev_weights.update(_load_safetensors(wf, device="cpu"))
        else:
            prev_weights = torch.load(
                os.path.join(prev_adapter_dir, "adapter_model.bin"),
                map_location="cpu",
            )
        set_peft_model_state_dict(model, prev_weights)
        del prev_weights
        print("  LoRA weights seeded from previous round.")

    # Loss callback
    loss_log = []

    from transformers import TrainerCallback

    class LossLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_log.append({"step": state.global_step, "loss": logs["loss"]})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,   # pre-mapped — no remapping here
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        dataset_num_proc=cfg.get("dataset_num_proc", 1),
        packing=cfg.get("packing", False),
        args=TrainingArguments(
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg.get("warmup_steps", 50),
            max_steps=steps,
            learning_rate=cfg["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim=cfg.get("optim", "adamw_8bit"),
            weight_decay=cfg.get("weight_decay", 0.1),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            seed=cfg.get("seed", 3407),
            output_dir=cfg.get("output_dir", "unsloth_australian_legal_lora"),
            save_strategy="no",
            disable_tqdm=False,
        ),
        callbacks=[LossLogger()],
    )

    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    # Save adapter
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"  Adapter saved → {adapter_dir}")

    # Free VRAM before returning
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return loss_log


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.list_configs:
        for key in GPU_CONFIGS:
            print(f"  --gpu {key}")
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    log_path = output_dir / "evolution_log.json"
    evolution_log = []

    print(f"\n{'='*60}")
    print(f"  Australian Law LLM — Recursive Self-Evolution")
    print(f"  GPU config    : {args.gpu}")
    print(f"  Rounds        : {args.rounds}")
    print(f"  Steps / round : {args.steps_per_round}")
    print(f"  LM Studio URL : {args.lm_studio_url}")
    print(f"  Output dir    : {output_dir}")
    print(f"{'='*60}\n")

    # Connect to LM Studio
    print("Connecting to LM Studio...")
    lm_client = get_lm_studio_client(args.lm_studio_url)
    lm_model  = args.lm_studio_model or detect_lm_studio_model(lm_client)

    # Load base config
    cfg = load_base_config(args.gpu)
    cfg["max_steps"] = args.steps_per_round

    # Load and map dataset ONCE — shared across all rounds, no remapping
    from unsloth import FastLanguageModel
    _init_model, _init_tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=cfg["dtype"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    shared_dataset = load_and_map_dataset(
        _init_tokenizer, cfg.get("token_truncation_length", 500)
    )
    del _init_model, _init_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    history = []
    prev_adapter_dir = None  # updated each round so the next round continues from here

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'─'*60}")
        print(f"  ROUND {round_num}/{args.rounds}")
        print(f"  Config: lr={cfg['learning_rate']}, r={cfg['r']}, "
              f"lora_alpha={cfg['lora_alpha']}, wd={cfg['weight_decay']}, "
              f"warmup={cfg.get('warmup_steps')}, batch={cfg['per_device_train_batch_size']}")
        if prev_adapter_dir:
            print(f"  Continuing from: {prev_adapter_dir}")
        print(f"{'─'*60}\n")

        adapter_dir = str(output_dir / f"round_{round_num:02d}_adapter")

        # ── Train (continuing from previous round's adapter) ──
        t0 = time.time()
        loss_log = run_training_round(cfg, args.steps_per_round, adapter_dir,
                                      shared_dataset, prev_adapter_dir)
        elapsed = time.time() - t0

        if not loss_log:
            print("  WARNING: no loss values captured — skipping LM Studio query")
            continue

        start_loss = loss_log[0]["loss"]
        end_loss   = loss_log[-1]["loss"]
        print(f"\n  Round {round_num} complete in {elapsed/60:.1f} min  "
              f"| loss: {start_loss:.4f} → {end_loss:.4f}")

        # ── Evaluate all 100 questions ──
        eval_results = evaluate_round(cfg, adapter_dir, output_dir, round_num)
        # Send every 5th answer (20 total) — representative coverage across all
        # legal areas without blowing the LM Studio context window.
        # Each answer capped at 600 chars so multi-line answers don't overflow.
        sample_answers = "\n\n".join(
            f"Q{r['q']:03d}: {r['question']}\nA: {r['answer'][:600]}{'…' if len(r['answer']) > 600 else ''}"
            for r in eval_results[::5]
        )

        # ── Ask LM Studio ──
        print("\n  Querying LM Studio for config improvements...")
        prompt   = build_lm_studio_prompt(round_num, cfg, loss_log, history,
                                          sample_answers)
        response = ask_lm_studio(lm_client, lm_model, prompt)
        print(f"  LM Studio response:\n    {response[:300].replace(chr(10), ' ')}")

        patch    = extract_json(response)
        new_cfg, changes = apply_patch(cfg, patch)

        if changes:
            print("  Applying changes:")
            for c in changes:
                print(c)
        else:
            print("  No config changes suggested.")

        # ── Record history ──
        record = {
            "round":        round_num,
            "timestamp":    datetime.now().isoformat(),
            "config":       {k: cfg[k] for k in TUNABLE_KEYS if k in cfg},
            "loss_log":     loss_log,
            "start_loss":   start_loss,
            "end_loss":     end_loss,
            "lm_response":  response,
            "patch":        patch,
            "changes":      changes,
            "adapter_dir":  adapter_dir,
            "eval_answers": str(output_dir / f"round_{round_num:02d}_answers.txt"),
        }
        evolution_log.append(record)
        history.append(record)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(evolution_log, f, indent=2, ensure_ascii=False)

        cfg = new_cfg
        cfg["max_steps"] = args.steps_per_round
        prev_adapter_dir = adapter_dir  # next round continues from this adapter

        # Clear dynamo recompilation cache between rounds to prevent progressive slowdown
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Self-evolution complete — {args.rounds} rounds finished")
    print(f"  Evolution log : {log_path}")
    print(f"  Best adapter  : round with lowest end loss")
    if evolution_log:
        best = min(evolution_log, key=lambda r: r["end_loss"])
        print(f"    → Round {best['round']}  end_loss={best['end_loss']:.4f}  "
              f"path={best['adapter_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
