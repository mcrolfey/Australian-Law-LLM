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
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


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
                            history: list[dict]) -> str:
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

    return f"""You are an expert machine learning engineer helping to optimise hyperparameters for a LLaMA LoRA fine-tuning run on Australian legal text using Unsloth on a Windows machine with an RTX 3070 (8GB VRAM).

TRAINING SUMMARY — Round {round_num}
======================================
Current config:
{current_config_str}

Loss curve this round:
{loss_summary}

Previous rounds:
{history_str}

TASK
====
Analyse the loss curve. If the loss is still decreasing steeply, suggest continuing with similar or slightly higher learning rate. If the loss has plateaued or is noisy, adjust accordingly.

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


def load_and_map_dataset(tokenizer, trunc_len: int):
    """Download (or use cache) and map the corpus. Called once before the loop."""
    from datasets import load_dataset
    import kagglehub

    print("Loading Open Australian Legal Corpus (once — reused every round)...")
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
# Training
# ──────────────────────────────────────────────────────────────────────────────
def run_training_round(cfg: dict, steps: int, adapter_dir: str, dataset) -> list[dict]:
    """
    Run one training round with a pre-mapped dataset.
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

    # LoRA
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

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'─'*60}")
        print(f"  ROUND {round_num}/{args.rounds}")
        print(f"  Config: lr={cfg['learning_rate']}, r={cfg['r']}, "
              f"lora_alpha={cfg['lora_alpha']}, wd={cfg['weight_decay']}, "
              f"warmup={cfg.get('warmup_steps')}, batch={cfg['per_device_train_batch_size']}")
        print(f"{'─'*60}\n")

        adapter_dir = str(output_dir / f"round_{round_num:02d}_adapter")

        # ── Train ──
        t0 = time.time()
        loss_log = run_training_round(cfg, args.steps_per_round, adapter_dir, shared_dataset)
        elapsed = time.time() - t0

        if not loss_log:
            print("  WARNING: no loss values captured — skipping LM Studio query")
            continue

        start_loss = loss_log[0]["loss"]
        end_loss   = loss_log[-1]["loss"]
        print(f"\n  Round {round_num} complete in {elapsed/60:.1f} min  "
              f"| loss: {start_loss:.4f} → {end_loss:.4f}")

        # ── Ask LM Studio ──
        print("\n  Querying LM Studio for config improvements...")
        prompt   = build_lm_studio_prompt(round_num, cfg, loss_log, history)
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
            "round":       round_num,
            "timestamp":   datetime.now().isoformat(),
            "config":      {k: cfg[k] for k in TUNABLE_KEYS if k in cfg},
            "loss_log":    loss_log,
            "start_loss":  start_loss,
            "end_loss":    end_loss,
            "lm_response": response,
            "patch":       patch,
            "changes":     changes,
            "adapter_dir": adapter_dir,
        }
        evolution_log.append(record)
        history.append(record)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(evolution_log, f, indent=2, ensure_ascii=False)

        cfg = new_cfg
        cfg["max_steps"] = args.steps_per_round

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
