"""
Australian Law LLM - SFT Fine-Tuning Script (Pipeline Step 2)
==============================================================
Trains the model to follow Q&A instructions on Australian law.

This is STEP 2 of the two-step pipeline:
  Step 1 (cpt_train.py) — Continued Pre-Training on raw legal text
                           → output: lora_cpt_law_model/
  Step 2 (this script)  — SFT on Q&A pairs, optionally starting from
                           the CPT-adapted weights instead of the base model
                           → output: lora_australian_law_model/

Usage:
    python train.py --gpu 8gb                          # train from base model
    python train.py --gpu 8gb --cpt-model lora_cpt_law_model  # train from CPT adapters (recommended)
    python train.py --gpu 16gb --steps 200
    python train.py --list-configs
"""

import os
import gc
import sys
import argparse
import importlib

# Suppress HF transfer (improves stability on Windows)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# WINDOWS IMPORT ORDER FIX:
# triton must be imported BEFORE torch so that torch._inductor finds triton
# already in sys.modules during its own initialisation. If torch runs first,
# its internal inductor setup happens without triton and unsloth later dies
# silently via os._exit(). This ordering mirrors the working debug test.
try:
    import triton
    from torch._inductor.runtime.hints import DeviceProperties
except Exception:
    pass

import torch
from unsloth import FastLanguageModel

CONFIGS = {
    "4gb":  "configs.gpu_4gb",
    "8gb":  "configs.gpu_8gb",
    "16gb": "configs.gpu_16gb",
    "24gb": "configs.gpu_24gb",
}

GPU_DESCRIPTIONS = {
    "4gb":  "GTX 1650 / RTX 3050 / low VRAM   → Llama-3.2-1B,  seq=256",
    "8gb":  "RTX 3070 / RTX 3060 / mid VRAM   → Llama-3.2-3B,  seq=512",
    "16gb": "RTX 4080 / RTX 3080 Ti / hi VRAM → Llama-3.2-3B,  seq=1024",
    "24gb": "RTX 3090 / RTX 4090 / max VRAM   → Llama-3.1-8B,  seq=2048",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM on the Open Australian Legal Corpus"
    )
    parser.add_argument(
        "--gpu",
        choices=list(CONFIGS.keys()),
        default="8gb",
        help="VRAM tier to use (default: 8gb)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps from config",
    )
    parser.add_argument(
        "--cpt-model",
        default=None,
        help="Path to CPT LoRA adapter directory from cpt_train.py (Step 1). "
             "When set, SFT starts from the domain-adapted weights instead of "
             "the raw base model, which reduces hallucination.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print available GPU configs and exit",
    )
    parser.add_argument(
        "--trajectory-alpha",
        type=float,
        default=None,
        help="Trajectory regularisation strength (default: from GPU config, usually 0.01). "
             "Set to 0.0 to disable. Higher values enforce smoother layer transitions "
             "at the cost of slower convergence.",
    )
    return parser.parse_args()


def load_config(gpu_tier: str):
    module = importlib.import_module(CONFIGS[gpu_tier])
    return module.CONFIG


def download_model_with_retry(model_name: str, max_retries: int = 20, wait: int = 15) -> str:
    import socket
    import time
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    socket.setdefaulttimeout(60)

    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("  hf_transfer enabled — using fast multi-threaded downloader")
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("  Tip: pip install hf_transfer  for 5x faster downloads")

    WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".ckpt", ".msgpack", ".h5")

    def weights_present(directory: str) -> bool:
        try:
            return any(f.endswith(WEIGHT_EXTENSIONS) for f in os.listdir(directory))
        except OSError:
            return False

    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"Downloading {model_name}  (attempt {attempt}/{max_retries}) ...")
            local_dir = snapshot_download(
                repo_id=model_name,
                repo_type="model",
                resume_download=True,
                local_files_only=False,
            )
            if not weights_present(local_dir):
                raise RuntimeError(
                    f"Snapshot directory has no weight files: {local_dir}\n"
                    "The previous download was incomplete. Retrying..."
                )
            print(f"Model ready at: {local_dir}\n")
            return local_dir
        except (RepositoryNotFoundError, EntryNotFoundError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            if attempt >= max_retries:
                print(f"Download failed after {max_retries} attempts: {e}")
                sys.exit(1)
            print(f"  Download interrupted ({type(e).__name__}: {e})")
            print(f"  Resuming in {wait}s ...  (attempt {attempt}/{max_retries})")
            time.sleep(wait)


def detect_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def suggest_config() -> str:
    vram = detect_vram_gb()
    if vram >= 20:
        return "24gb"
    elif vram >= 12:
        return "16gb"
    elif vram >= 6:
        return "8gb"
    else:
        return "4gb"


def main():
    args = parse_args()

    if args.list_configs:
        print("\nAvailable GPU configs:\n")
        for key, desc in GPU_DESCRIPTIONS.items():
            print(f"  --gpu {key:5s}  {desc}")
        vram = detect_vram_gb()
        if vram > 0:
            suggestion = suggest_config()
            print(f"\nDetected VRAM: {vram:.1f} GB  →  Suggested config: --gpu {suggestion}\n")
        else:
            print("\nCould not detect GPU VRAM — specify --gpu manually based on the table above.\n")
        sys.exit(0)

    cfg = load_config(args.gpu)
    if args.steps is not None:
        cfg["max_steps"] = args.steps
    if args.trajectory_alpha is not None:
        cfg["trajectory_alpha"] = args.trajectory_alpha

    print(f"\n{'='*56}")
    print(f"  Australian Law LLM Fine-Tuning")
    print(f"  GPU config : --gpu {args.gpu}  ({GPU_DESCRIPTIONS[args.gpu]})")
    print(f"  Model      : {cfg['model_name']}")
    print(f"  Seq length : {cfg['max_seq_length']}")
    print(f"  Steps      : {cfg['max_steps']}")
    print(f"{'='*56}\n")

    # --------------------------------------------------
    # 0. VRAM Pre-Clean
    # --------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    # --------------------------------------------------
    # 1. Load Model
    # --------------------------------------------------
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trajectory_trainer import TrajectoryTrainer

    # --------------------------------------------------
    # 1a. Determine base model — either raw HF model or CPT-adapted weights
    # --------------------------------------------------
    if args.cpt_model:
        # Load the CPT-adapted base model by merging the CPT LoRA into memory,
        # then attach a fresh set of SFT LoRA adapters on top.
        # We use the base model name from the CPT adapter config so the right
        # architecture is loaded.
        import json
        cpt_cfg_path = os.path.join(args.cpt_model, "adapter_config.json")
        if not os.path.exists(cpt_cfg_path):
            print(f"ERROR: No adapter_config.json found in '{args.cpt_model}'.")
            print("Run cpt_train.py first to generate the CPT adapters.")
            sys.exit(1)
        with open(cpt_cfg_path) as f:
            cpt_adapter_cfg = json.load(f)
        base_model_name = cpt_adapter_cfg["base_model_name_or_path"]
        print(f"Loading CPT-adapted model: {base_model_name} + {args.cpt_model}")
        local_cpt_base = download_model_with_retry(base_model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=local_cpt_base,
            max_seq_length=cfg["max_seq_length"],
            dtype=cfg["dtype"],
            load_in_4bit=cfg["load_in_4bit"],
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.cpt_model)
        # Merge the CPT weights into the base so the new SFT LoRA sits on top
        # of the domain-adapted weights, not alongside them.
        model = model.merge_and_unload()
        print("CPT adapters merged. Attaching fresh SFT LoRA adapters...")
    else:
        local_model_path = download_model_with_retry(cfg["model_name"])
        print(f"Loading {cfg['model_name']} in 4-bit...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=local_model_path,
            max_seq_length=cfg["max_seq_length"],
            dtype=cfg["dtype"],
            load_in_4bit=cfg["load_in_4bit"],
        )
    tokenizer.padding_side = "right"

    # --------------------------------------------------
    # 2. Inject LoRA Adapters
    # --------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    # --------------------------------------------------
    # 3. Load dataset
    # --------------------------------------------------
    # Prefer a generated Q&A dataset (qa_dataset.jsonl) over the raw corpus.
    # Q&A format aligns training with the benchmark test format — the model
    # learns to answer questions rather than complete documents.
    # Generate qa_dataset.jsonl first with: python generate_qa.py
    QA_DATASET = "qa_dataset.jsonl"
    KAGGLE_CACHE = os.path.expanduser(
        "~/.cache/kagglehub/datasets/umarbutler/open-australian-legal-corpus/versions/2/corpus.jsonl"
    )

    EOS_TOKEN = tokenizer.eos_token

    if os.path.exists(QA_DATASET):
        print(f"Loading Q&A dataset: {QA_DATASET}")
        dataset = load_dataset("json", data_files=QA_DATASET, split="train")
        print(f"  {len(dataset):,} Q&A pairs loaded")

        qa_prompt = """\
Below is a question about Australian law. Write a response that accurately and concisely answers it.

### Instruction:
You are an expert Australian legal assistant trained on the Open Australian Legal Corpus. \
Answer the following question accurately, citing relevant legislation or case law where appropriate.

### Input:
{question}

### Response:
{answer}"""

        def format_qa(examples):
            return {"text": [
                qa_prompt.format(question=q, answer=a) + EOS_TOKEN
                for q, a in zip(examples["question"], examples["answer"])
            ]}

        print("Formatting Q&A dataset...")
        dataset = dataset.map(format_qa, batched=True)

    else:
        # Fall back to raw corpus (document completion) if Q&A dataset not yet generated.
        # Run: python generate_qa.py   to create qa_dataset.jsonl first.
        print(f"NOTE: {QA_DATASET} not found — falling back to raw corpus (document completion).")
        print("      Run 'python generate_qa.py' to generate a Q&A dataset for better results.\n")

        if os.path.exists(KAGGLE_CACHE):
            corpus_path = KAGGLE_CACHE
            print(f"  Using cached corpus: {corpus_path}")
        else:
            import kagglehub
            dataset_dir = kagglehub.dataset_download("umarbutler/open-australian-legal-corpus")
            corpus_path = os.path.join(dataset_dir, "corpus.jsonl")

        dataset = load_dataset("json", data_files=corpus_path, split="train")

        alpaca_prompt = """\
Below is an instruction that describes a legal task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the following Australian legal document. Synthesize its legal principles based on the provided jurisdiction and context framework.

### Input:
Citation: {}
Jurisdiction: {}
Document Type: {}

### Response:
{}"""

        trunc_len = cfg["token_truncation_length"]

        def formatting_prompts_func(examples):
            citations     = examples.get("citation",     ["Unknown Citation"]     * len(examples["text"]))
            jurisdictions = examples.get("jurisdiction", ["Unknown Jurisdiction"] * len(examples["text"]))
            types         = examples.get("type",         ["Unknown Type"]         * len(examples["text"]))
            texts         = examples["text"]
            formatted = []
            for citation, jurisdiction, doc_type, text in zip(citations, jurisdictions, types, texts):
                raw = alpaca_prompt.format(citation, jurisdiction, doc_type, text or "") + EOS_TOKEN
                tokens = tokenizer.encode(raw, truncation=True, max_length=trunc_len)
                formatted.append(tokenizer.decode(tokens, skip_special_tokens=False))
            return {"text": formatted}

        print("Formatting dataset...")
        dataset = dataset.map(formatting_prompts_func, batched=True)

    # --------------------------------------------------
    # 4. Trainer
    # --------------------------------------------------
    trajectory_alpha = cfg.get("trajectory_alpha", 0.01)
    print(f"  Trajectory regularisation: alpha={trajectory_alpha}"
          f"{'  (disabled)' if trajectory_alpha == 0.0 else ''}")

    trainer = TrajectoryTrainer(
        trajectory_alpha=trajectory_alpha,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        dataset_num_proc=cfg["dataset_num_proc"],
        packing=cfg["packing"],
        args=TrainingArguments(
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg["warmup_steps"],
            max_steps=cfg["max_steps"],
            learning_rate=cfg["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=cfg.get("logging_steps", 1),
            optim=cfg["optim"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            seed=cfg["seed"],
            output_dir=cfg["output_dir"],
            # gradient_checkpointing trades recomputation for VRAM — important
            # when output_hidden_states=True retains all intermediate activations.
            gradient_checkpointing=True,
            # Windows pickle bug: torch.save(SFTConfig) fails mid-training.
            # Intermediate checkpoints are disabled; adapters save at the end.
            save_strategy="no",
            disable_tqdm=False,
        ),
    )

    # --------------------------------------------------
    # 5. Train & Save
    # --------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    print("Starting fine-tuning...\n")
    trainer.train()

    save_dir = cfg["lora_save_dir"]
    print(f"\nSaving LoRA adapters to '{save_dir}'...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"\n{'='*56}")
    print(f"  SUCCESS! Adapters saved to: '{save_dir}'")
    print(f"  Run the web UI:  python serve.py --model {save_dir}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
