"""
Quick command-line test for the fine-tuned model.
Usage: python test_model.py "What are the elements of negligence in Australian law?"
       python test_model.py   # interactive mode
"""

import sys
import os

try:
    import triton
    from torch._inductor.runtime.hints import DeviceProperties
except Exception:
    pass

import torch
from unsloth import FastLanguageModel

MODEL_DIR = "lora_australian_law_model"
MAX_NEW_TOKENS = 512

def load():
    import json
    with open(os.path.join(MODEL_DIR, "adapter_config.json")) as f:
        base_model = json.load(f)["base_model_name_or_path"]
    print(f"Loading {base_model} + LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    print("Ready.\n")
    return model, tokenizer

def ask(model, tokenizer, question):
    prompt = f"""Below is an instruction that describes a legal task. Write a response that appropriately completes the request.

### Instruction:
You are an expert Australian legal assistant. Answer the following question accurately.

### Input:
{question}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=0.2,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def main():
    model, tokenizer = load()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Q: {question}\n")
        print(f"A: {ask(model, tokenizer, question)}\n")
    else:
        print("Interactive mode — type your question, or 'quit' to exit.\n")
        while True:
            question = input("Q: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if question:
                print(f"\nA: {ask(model, tokenizer, question)}\n")

if __name__ == "__main__":
    main()
