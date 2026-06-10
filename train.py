"""
Australian Law LLM - Fine-Tuning Script
========================================
Fine-tunes a LLaMA model on the Open Australian Legal Corpus using Unsloth.

Usage:
    python train.py --gpu 8gb          # pick a config tier
    python train.py --gpu 16gb --steps 200
    python train.py --list-configs     # show available GPU configs
"""

import os
import gc
import sys
import argparse
import importlib
import torch

# Pre-warm triton and torch._inductor before unsloth is imported.
# On Windows, unsloth crashes silently if these are not already cached
# in sys.modules when it loads. This is a known Windows/triton ordering issue.
try:
    import triton
    from torch._inductor.runtime.hints import DeviceProperties
except Exception:
    pass

# Suppress HF transfer (improves stability on Windows)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

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
        "--list-configs",
        action="store_true",
        help="Print available GPU configs and exit",
    )
    return parser.parse_args()


def load_config(gpu_tier: str):
    module = importlib.import_module(CONFIGS[gpu_tier])
    return module.CONFIG


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
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"Loading {cfg['model_name']} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
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
    # 3. Load Open Australian Legal Corpus
    # --------------------------------------------------
    print("Downloading Open Australian Legal Corpus via kagglehub...")
    import kagglehub
    dataset_dir = kagglehub.dataset_download("umarbutler/open-australian-legal-corpus")
    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    print(f"Dataset path: {corpus_path}")

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

    EOS_TOKEN = tokenizer.eos_token
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
    trainer = SFTTrainer(
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
            logging_steps=1,
            optim=cfg["optim"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            seed=cfg["seed"],
            output_dir=cfg["output_dir"],
            save_strategy="no",   # Windows: avoids pickle errors during training
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
