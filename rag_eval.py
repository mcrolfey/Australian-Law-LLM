"""
Australian Law LLM - RAG / Open-Book Evaluation Script
=======================================================
Tests the fine-tuned model with retrieved legal context supplied at inference
time, preventing reliance on hallucinated parametric memory.

Input file format (JSON or CSV):
  JSON: [{"question": "...", "context": "..."}, ...]
  CSV:  columns named 'question' and 'context'

Usage:
    python rag_eval.py --input test_data.json
    python rag_eval.py --input test_data.csv --output my_results.txt
    python rag_eval.py --input test_data.json --base-only
    python rag_eval.py --input test_data.json --start 10   # resume
"""

import argparse
import json
import os
import sys
import time

try:
    import triton
    from torch._inductor.runtime.hints import DeviceProperties
except Exception:
    pass

import torch
from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """You are a specialized Australian Legal Research Assistant.
Your task is to answer the user's question using ONLY the provided Legal Context.

CRITICAL CONSTRAINTS:
1. If the provided context does not contain the answer, you MUST state: "I cannot find the answer in the provided context."
2. DO NOT use your own external knowledge or training data to fill in gaps.
3. If the context is incomplete or nonsensical, do not attempt to fabricate information.
4. If you identify a list of sections or provisions in the context that do not answer the question, do not repeat them.
5. Keep your answer concise and strictly grounded in the text provided.

LEGAL CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def parse_args():
    parser = argparse.ArgumentParser(description="RAG evaluation for Australian Law LLM")
    parser.add_argument("--input", required=True,
                        help="Path to JSON or CSV file with 'question' and 'context' columns")
    parser.add_argument("--model", default="lora_australian_law_model",
                        help="LoRA adapter directory (default: lora_australian_law_model)")
    parser.add_argument("--base-only", action="store_true",
                        help="Load base model only — no LoRA adapters")
    parser.add_argument("--base-model", default=None,
                        help="Base model name (auto-detected from adapter config if omitted)")
    parser.add_argument("--output", default=None,
                        help="Output text file (default: rag_batch_results.txt)")
    parser.add_argument("--json-output", default=None,
                        help="Output JSON file (default: rag_batch_results.json)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--start", type=int, default=1,
                        help="Resume from row N (1-indexed)")
    return parser.parse_args()


def load_dataset(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Support {"rows": [...]} envelope
            data = next(iter(data.values()))
    elif ext == ".csv":
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .json or .csv")

    # Validate required columns
    for i, row in enumerate(data):
        for col in ("question", "context"):
            if col not in row:
                raise ValueError(f"Row {i+1} is missing required column '{col}'")
    return data


def load_model(args):
    if args.base_only:
        base_model = args.base_model
        if not base_model:
            adapter_cfg = os.path.join(args.model, "adapter_config.json")
            if os.path.exists(adapter_cfg):
                with open(adapter_cfg) as f:
                    base_model = json.load(f)["base_model_name_or_path"]
                print(f"Auto-detected base model: {base_model}")
            else:
                raise ValueError("--base-only requires --base-model <name>")
        print(f"Loading BASE model: {base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        label = f"BASE: {base_model}"
    else:
        adapter_cfg = os.path.join(args.model, "adapter_config.json")
        if not os.path.exists(adapter_cfg):
            print(f"ERROR: No adapter_config.json found in '{args.model}'.")
            print("Run train.py first, or pass --base-only to test the base model.")
            sys.exit(1)
        with open(adapter_cfg) as f:
            base_model = args.base_model or json.load(f)["base_model_name_or_path"]
        print(f"Loading fine-tuned model: {base_model} + LoRA {args.model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model)
        label = f"FINE-TUNED: {args.model}"

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    print(f"Model ready: {label}\n")
    return model, tokenizer, label


def generate(model, tokenizer, question: str, context: str, max_new_tokens: int) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question, context=context.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.1,         # very low — forces context-grounded, deterministic output
            repetition_penalty=1.15, # prevents infinite token loops
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    args = parse_args()

    txt_out  = args.output      or "rag_batch_results.txt"
    json_out = args.json_output or "rag_batch_results.json"

    print(f"Loading dataset: {args.input}")
    rows = load_dataset(args.input)
    total = len(rows)
    print(f"  {total} rows loaded\n")

    model, tokenizer, label = load_model(args)

    results = []
    if args.start > 1 and os.path.exists(json_out):
        with open(json_out, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from row {args.start} ({len(results)} already done)\n")

    print(f"Running {total - (args.start - 1)} rows — output: {txt_out}\n")
    print("=" * 70)

    for i, row in enumerate(rows, start=1):
        if i < args.start:
            continue

        question = row["question"].strip()
        context  = row["context"].strip()

        print(f"\n[{i}/{total}] {question[:80]}{'...' if len(question) > 80 else ''}")
        t0 = time.time()
        answer = generate(model, tokenizer, question, context, args.max_tokens)
        elapsed = time.time() - t0
        print(f"  → {len(answer.split())} words  ({elapsed:.1f}s)")

        results.append({
            "row": i,
            "question": question,
            "context_length": len(context.split()),
            "answer": answer,
        })

        # Save after every row so progress is never lost
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(txt_out, "w", encoding="utf-8") as f:
            f.write("AUSTRALIAN LAW LLM — RAG EVALUATION RESULTS\n")
            f.write(f"Model  : {label}\n")
            f.write(f"Input  : {args.input}\n")
            f.write(f"Rows answered: {len(results)}/{total}\n")
            f.write("=" * 70 + "\n\n")
            for r in results:
                f.write(f"Row {r['row']:03d}. {r['question']}\n")
                f.write(f"[Context: {r['context_length']} words]\n")
                f.write("-" * 70 + "\n")
                f.write(r["answer"] + "\n\n")

    print(f"\n{'='*70}")
    print(f"  Done! {len(results)} rows answered.")
    print(f"  Readable results : {txt_out}")
    print(f"  JSON results     : {json_out}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
