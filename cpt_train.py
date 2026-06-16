"""
Australian Law LLM - Continued Pre-Training (CPT) / Domain Adaptation
=======================================================================
PIPELINE STEP 1 OF 2

Purpose
-------
Teach the base LLaMA model the *factual content* of Australian law before
any instruction-following is trained. This dramatically reduces hallucination
in the downstream SFT step by ensuring the model has seen the actual text of
Acts, regulations, and judgments during training.

Training objective: Causal Language Modelling (next-token prediction) on raw
legal text. No instruction template — just packed, dense legal text.

Pipeline:
  [Step 1 — this script]
      Base model  ──CPT──►  lora_cpt_law_model/   (factual knowledge)
  [Step 2 — train.py]
      lora_cpt_law_model/  ──SFT──►  lora_australian_law_model/  (Q&A skill)

Usage:
    python cpt_train.py --gpu 8gb
    python cpt_train.py --gpu 8gb --data-dir ./law_texts
    python cpt_train.py --gpu 8gb --steps 500
    python cpt_train.py --list-configs

Input data format (see DATA PREPARATION section at the bottom of this file):
    Option A — .jsonl files where every line has a "text" key:
        {"text": "Corporations Act 2001 (Cth)\\n\\nPart 1.1 — Preliminary ..."}
    Option B — plain .txt files (one document per file or one per line)
    Option C — Kaggle Open Australian Legal Corpus (default, no --data-dir needed)
"""

import gc
import importlib
import os
import sys
import argparse

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# WINDOWS IMPORT ORDER FIX — triton must be in sys.modules before torch loads
# its inductor subsystem. Unsloth will silently os._exit() if this is wrong.
try:
    import triton
    from torch._inductor.runtime.hints import DeviceProperties
except Exception:
    pass

import torch
from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# GPU Tier Configs for CPT
# ---------------------------------------------------------------------------
# CPT benefits from longer sequences (more context per document) and higher
# LoRA rank (deeper factual injection into FFN weights) vs SFT.
# Sequence lengths here are 4–16× longer than the SFT configs.

CPT_CONFIGS = {
    "4gb": {
        "model_name":    "unsloth/Llama-3.2-1B-bnb-4bit",   # base, not instruct
        "max_seq_length": 2048,
        "r":              32,
        "lora_alpha":     32,
        "batch_size":     1,
        "grad_accum":     16,
        "warmup_steps":   10,
        "max_steps":      200,
        "learning_rate":  5e-5,
        "description":    "GTX 1650 / RTX 3050  →  Llama-3.2-1B,  seq=2048",
    },
    "8gb": {
        # Laptop 8 GB GPUs lose ~1 GB to the display driver, leaving ~4 GB
        # free after the model loads. The 1B model (~1.5 GB) gives enough
        # headroom for the fused CE loss; the 3B model (~3 GB) does not.
        # CPT injects legal knowledge via volume of text, not model size —
        # the 1B CPT adapters are then used to bootstrap the 3B SFT run.
        "model_name":    "unsloth/Llama-3.2-1B-bnb-4bit",
        "max_seq_length": 2048,
        "r":              32,
        "lora_alpha":     32,
        "batch_size":     1,
        "grad_accum":     16,
        "warmup_steps":   10,
        "max_steps":      300,
        "learning_rate":  5e-5,
        "description":    "RTX 3070 / RTX 3060  →  Llama-3.2-1B,  seq=2048 (display VRAM constraint)",
    },
    "16gb": {
        "model_name":    "unsloth/Llama-3.2-3B-bnb-4bit",
        "max_seq_length": 8192,
        "r":              64,
        "lora_alpha":     64,
        "batch_size":     2,
        "grad_accum":     4,
        "warmup_steps":   10,
        "max_steps":      500,
        "learning_rate":  5e-5,
        "description":    "RTX 4080 / RTX 3080 Ti  →  Llama-3.2-3B,  seq=8192",
    },
    "24gb": {
        "model_name":    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "max_seq_length": 8192,
        "r":              64,
        "lora_alpha":     64,
        "batch_size":     2,
        "grad_accum":     4,
        "warmup_steps":   10,
        "max_steps":      500,
        "learning_rate":  3e-5,
        "description":    "RTX 3090 / RTX 4090  →  Llama-3.1-8B,   seq=8192",
    },
}

# All linear projection modules — targeting FFN (gate/up/down) is critical for
# CPT because factual knowledge lives primarily in the feed-forward weights.
CPT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

CPT_OUTPUT_DIR = "lora_cpt_law_model"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CPT Step 1: Domain-adapt LLaMA to Australian law")
    p.add_argument("--gpu", choices=list(CPT_CONFIGS), default="8gb",
                   help="VRAM tier (default: 8gb)")
    p.add_argument("--steps", type=int, default=None,
                   help="Override max_steps from config")
    p.add_argument("--data-dir", default=None,
                   help="Directory containing .jsonl or .txt files. "
                        "Omit to use the Kaggle Open Australian Legal Corpus.")
    p.add_argument("--output", default=CPT_OUTPUT_DIR,
                   help=f"Where to save LoRA adapters (default: {CPT_OUTPUT_DIR})")
    p.add_argument("--list-configs", action="store_true",
                   help="Print GPU configs and exit")
    return p.parse_args()


def detect_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


# ---------------------------------------------------------------------------
# Robust model download — retries until every shard lands
# ---------------------------------------------------------------------------

def download_model_with_retry(model_name: str, max_retries: int = 20, wait: int = 15) -> str:
    """
    Download a HuggingFace model with resume support and automatic retry.

    Two layers of protection against stalled downloads:
      1. socket.setdefaulttimeout — kills TCP connections that go silent for
         60 s so the retry loop actually fires instead of hanging forever.
      2. hf_transfer — if installed, uses a multi-threaded Rust downloader
         that is 5-10x faster and far less likely to stall than the Python one.
         Install with: pip install hf_transfer

    huggingface_hub stores partial shards as *.incomplete files and resumes
    from the byte offset where the last attempt stopped.
    """
    import socket
    import time
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    # Kill stalled TCP connections after 60 s of silence.
    # Without this, a hung shard sits at 0 B/s forever and the retry loop
    # never gets a chance to fire.
    socket.setdefaulttimeout(60)

    # Enable hf_transfer fast downloader if available
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
            return any(
                f.endswith(WEIGHT_EXTENSIONS)
                for f in os.listdir(directory)
            )
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
            # Verify weight files actually landed — a previous interrupted
            # download can leave a snapshot directory with only config/tokenizer
            # files and no weights.
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_corpus_from_dir(data_dir: str, tokenizer):
    """
    Load raw legal text from a directory of .jsonl or .txt files.

    .jsonl — expects a "text" field per line:
        {"text": "Corporations Act 2001 (Cth)..."}

    .txt — treated as one document per file. Each file becomes one row.
    """
    from datasets import load_dataset, concatenate_datasets, Dataset

    jsonl_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".jsonl")
    ]
    txt_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ]

    parts = []

    if jsonl_files:
        print(f"  Loading {len(jsonl_files)} .jsonl file(s)...")
        ds = load_dataset("json", data_files=jsonl_files, split="train")
        # Keep only the text column; rename if needed
        if "text" not in ds.column_names:
            raise ValueError(
                "No 'text' column found in .jsonl files. "
                "Each line must be: {\"text\": \"...legal content...\"}"
            )
        parts.append(ds.select_columns(["text"]))

    if txt_files:
        print(f"  Loading {len(txt_files)} .txt file(s)...")
        rows = []
        for path in txt_files:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
            if content:
                rows.append({"text": content})
        if rows:
            parts.append(Dataset.from_list(rows))

    if not parts:
        raise FileNotFoundError(
            f"No .jsonl or .txt files found in '{data_dir}'. "
            "See the DATA PREPARATION section at the bottom of cpt_train.py."
        )

    from datasets import concatenate_datasets
    dataset = concatenate_datasets(parts)
    print(f"  Total documents loaded: {len(dataset):,}")
    return dataset


def load_kaggle_corpus():
    """
    Load the Open Australian Legal Corpus.

    Checks the kagglehub local cache first so the script works offline if the
    dataset was previously downloaded. Only contacts Kaggle if not cached.
    """
    from datasets import load_dataset

    KAGGLE_SLUG = "umarbutler/open-australian-legal-corpus"
    CACHE_BASE  = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub",
                               "datasets", KAGGLE_SLUG.replace("/", os.sep),
                               "versions")

    corpus_path = None

    # Walk the versioned cache directories newest-first
    if os.path.isdir(CACHE_BASE):
        versions = sorted(
            (v for v in os.listdir(CACHE_BASE) if v.isdigit()),
            key=int, reverse=True
        )
        for ver in versions:
            candidate = os.path.join(CACHE_BASE, ver, "corpus.jsonl")
            if os.path.isfile(candidate):
                corpus_path = candidate
                print(f"Using cached corpus: {corpus_path}")
                break

    if corpus_path is None:
        print("Corpus not found in cache — downloading via kagglehub...")
        import kagglehub
        dataset_dir = kagglehub.dataset_download(KAGGLE_SLUG)
        corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
        print(f"Dataset path: {corpus_path}")

    dataset = load_dataset("json", data_files=corpus_path, split="train")
    print(f"Total documents: {len(dataset):,}")
    return dataset


def prepare_text_column(dataset, tokenizer):
    """
    Ensure every row has a "text" field and append EOS so the packer knows
    where each document ends. No instruction template — raw text only.
    """
    EOS = tokenizer.eos_token

    def add_eos(batch):
        return {"text": [t.strip() + EOS for t in batch["text"] if t and t.strip()]}

    dataset = dataset.map(add_eos, batched=True, remove_columns=[
        c for c in dataset.column_names if c != "text"
    ])

    # Drop empty rows
    dataset = dataset.filter(lambda x: len(x["text"]) > 10)
    print(f"Documents after cleaning: {len(dataset):,}")
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.list_configs:
        print("\nCPT GPU configs:\n")
        for key, cfg in CPT_CONFIGS.items():
            print(f"  --gpu {key:5s}  {cfg['description']}")
        vram = detect_vram_gb()
        if vram > 0:
            rec = "24gb" if vram >= 20 else "16gb" if vram >= 12 else "8gb" if vram >= 6 else "4gb"
            print(f"\nDetected VRAM: {vram:.1f} GB  →  Suggested: --gpu {rec}\n")
        sys.exit(0)

    cfg = CPT_CONFIGS[args.gpu]
    if args.steps is not None:
        cfg = dict(cfg)           # copy so we don't mutate the constant
        cfg["max_steps"] = args.steps

    print(f"\n{'='*60}")
    print(f"  Australian Law LLM — CPT (Step 1 of 2)")
    print(f"  GPU config  : --gpu {args.gpu}  ({cfg['description']})")
    print(f"  Model       : {cfg['model_name']}")
    print(f"  Seq length  : {cfg['max_seq_length']}")
    print(f"  LoRA rank   : {cfg['r']}")
    print(f"  Steps       : {cfg['max_steps']}")
    print(f"  Output      : {args.output}")
    print(f"{'='*60}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # Defer heavy imports — keeps --list-configs fast and preserves the
    # triton→torch module-cache ordering on Windows.
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ------------------------------------------------------------------
    # 1. Download model (resume-safe, retries on timeout)
    # ------------------------------------------------------------------
    # We download first via huggingface_hub — it resumes interrupted shards
    # automatically. Then we pass the local cache path to Unsloth so it
    # never tries to hit the network again.
    local_model_path = download_model_with_retry(cfg["model_name"])

    print(f"Loading {cfg['model_name']} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    # CPT packs sequences left-to-right; padding_side doesn't matter much
    # but right is conventional for causal LM packing.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. LoRA — target ALL linear modules for deep knowledge injection
    # ------------------------------------------------------------------
    # Higher rank (64 vs 16 for SFT) injects more knowledge capacity.
    # use_rslora=True scales alpha by sqrt(r), stabilising training at r=64.
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["r"],
        target_modules=CPT_TARGET_MODULES,
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,          # 0 is recommended by Unsloth for speed
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,         # rank-stabilised LoRA — recommended for r >= 32
    )

    # ------------------------------------------------------------------
    # 3. Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    if args.data_dir:
        dataset = load_corpus_from_dir(args.data_dir, tokenizer)
    else:
        dataset = load_kaggle_corpus()

    dataset = prepare_text_column(dataset, tokenizer)

    # ------------------------------------------------------------------
    # 4. Trainer — packing=True is the key difference from SFT
    # ------------------------------------------------------------------
    # packing=True concatenates documents and slices them into fixed-length
    # chunks of max_seq_length. This eliminates padding waste and means every
    # token in every batch is a real training signal — far more efficient for
    # dense pre-training on raw text than SFT-style per-example batches.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        dataset_num_proc=1,      # >1 causes multiprocessing errors on Windows
        packing=True,            # ← causal LM packing — critical for CPT
        args=TrainingArguments(
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            warmup_steps=cfg["warmup_steps"],
            max_steps=cfg["max_steps"],
            learning_rate=cfg["learning_rate"],
            # Lower LR than SFT (5e-5 vs 2e-4) to avoid catastrophic
            # forgetting of the model's original language capabilities.
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",   # cosine decay suits long CPT runs
            seed=3407,
            output_dir="cpt_checkpoints",
            save_strategy="no",           # avoids Windows pickle errors
            disable_tqdm=False,
        ),
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU: {gpu_stats.name}  |  VRAM: {max_memory} GB  |  Reserved: {start_gpu_memory} GB")
    print("Starting CPT training...\n")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"\nPeak VRAM used: {used_memory} GB")
    print(f"Training time : {trainer_stats.metrics['train_runtime']:.0f}s")

    # ------------------------------------------------------------------
    # 6. Save CPT adapters
    # ------------------------------------------------------------------
    print(f"\nSaving CPT LoRA adapters to '{args.output}'...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\n{'='*60}")
    print(f"  CPT COMPLETE — adapters saved to: '{args.output}'")
    print(f"")
    print(f"  NEXT STEP: Run SFT (Step 2) using this as the base model:")
    print(f"    python train.py --gpu {args.gpu} --cpt-model {args.output}")
    print(f"  Or manually pass the adapter path to train.py when prompted.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


# =============================================================================
# DATA PREPARATION
# =============================================================================
#
# Option A — JSONL (recommended)
# --------------------------------
# One JSON object per line, with a "text" key containing the full document.
# Ideal for AustLII bulk downloads or pre-processed corpus extracts.
#
#   {"text": "Corporations Act 2001 (Cth)\n\nPart 1.1 — Preliminary\n\n1 Short title\n  This Act may be cited as the Corporations Act 2001."}
#   {"text": "Mabo v Queensland (No 2) [1992] HCA 23\n\nBrennan J:..."}
#
# One file can have millions of lines; multiple files in the same directory
# are all loaded and concatenated automatically.
#
#
# Option B — Plain .txt files
# ----------------------------
# One .txt file per document (or per batch of documents). The entire file
# is treated as one training example before being packed by SFTTrainer.
#
#   law_texts/
#     corporations_act_2001.txt
#     mabo_v_queensland_1992.txt
#     competition_consumer_act_2010.txt
#
#
# Option C — Default (no --data-dir)
# ------------------------------------
# Uses the Open Australian Legal Corpus from Kaggle automatically.
# ~202,000 documents covering all Australian jurisdictions.
# Requires ~/.kaggle/kaggle.json (see README).
#
#
# Recommended sources for additional CPT data
# ---------------------------------------------
# • AustLII bulk data export: https://www.austlii.edu.au/austlii/download.html
# • Federal Register of Legislation: https://www.legislation.gov.au
# • High Court of Australia decisions: https://www.hcourt.gov.au/cases/recent-decisions
