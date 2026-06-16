"""
Australian Law LLM - Batch Question Test
==========================================
Runs a set of legal questions through the fine-tuned OR base model and saves results.

Usage:
    # Fine-tuned model (default)
    python batch_test.py
    python batch_test.py --model lora_australian_law_model

    # Base model only (no LoRA — compare against fine-tuned)
    python batch_test.py --base-only --base-model unsloth/Llama-3.2-3B-Instruct

    # Options
    python batch_test.py --output results.txt --max-tokens 600
    python batch_test.py --start 42           # resume from question 42
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

PROMPT_TEMPLATE = """\
Below is an instruction that describes a legal task. Write a response that appropriately completes the request.

### Instruction:
You are an expert Australian legal assistant trained on the Open Australian Legal Corpus. \
Answer the following question accurately, citing relevant legislation or case law where appropriate.

### Input:
{question}

### Response:
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test the Australian Law LLM")
    parser.add_argument("--model", default="lora_australian_law_model",
                        help="Path to LoRA adapter directory (default: lora_australian_law_model)")
    parser.add_argument("--base-only", action="store_true",
                        help="Load the base model only — no LoRA adapters applied")
    parser.add_argument("--base-model", default=None,
                        help="Base model name/path (auto-detected from adapter config, or required with --base-only)")
    parser.add_argument("--output", default=None,
                        help="Output text file (default: batch_results_finetuned.txt or batch_results_base.txt)")
    parser.add_argument("--json-output", default=None,
                        help="Output JSON file (default: batch_results_finetuned.json or batch_results_base.json)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--start", type=int, default=1,
                        help="Start from question number (1-indexed, for resuming)")
    return parser.parse_args()


def load_model(args):
    import json

    if args.base_only:
        base_model = args.base_model
        if not base_model:
            # Try to read from adapter config anyway as a convenience
            adapter_cfg = os.path.join(args.model, "adapter_config.json")
            if os.path.exists(adapter_cfg):
                with open(adapter_cfg) as f:
                    base_model = json.load(f)["base_model_name_or_path"]
                print(f"Auto-detected base model from adapter config: {base_model}")
            else:
                raise ValueError(
                    "--base-only requires --base-model <name> (e.g. unsloth/Llama-3.2-3B-Instruct)"
                )
        print(f"Loading BASE model only: {base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        label = f"BASE: {base_model}"
    else:
        with open(os.path.join(args.model, "adapter_config.json")) as f:
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


def ask(model, tokenizer, question, max_new_tokens):
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.2,          # deterministic legal analysis, suppresses hallucination
            repetition_penalty=1.15,  # breaks looping/mode collapse patterns
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    args = parse_args()

    # Default output filenames based on mode
    suffix = "base" if args.base_only else "finetuned"
    txt_out  = args.output      or f"batch_results_{suffix}.txt"
    json_out = args.json_output or f"batch_results_{suffix}.json"

    model, tokenizer, label = load_model(args)

    total = len(QUESTIONS)
    results = []

    # Load existing JSON results if resuming
    if args.start > 1 and os.path.exists(json_out):
        with open(json_out) as f:
            results = json.load(f)
        print(f"Resuming from question {args.start} ({len(results)} already done)\n")

    print(f"Running {total - (args.start - 1)} questions — output: {txt_out}\n")
    print("=" * 70)

    for i, question in enumerate(QUESTIONS, start=1):
        if i < args.start:
            continue

        print(f"\n[{i}/{total}] {question[:80]}{'...' if len(question) > 80 else ''}")
        t0 = time.time()
        answer = ask(model, tokenizer, question, args.max_tokens)
        elapsed = time.time() - t0
        print(f"  → {len(answer.split())} words  ({elapsed:.1f}s)")

        results.append({"q": i, "question": question, "answer": answer})

        # Save after every question so progress isn't lost
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Write readable text file
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write("AUSTRALIAN LAW LLM — BATCH TEST RESULTS\n")
            f.write(f"Model: {label}\n")
            f.write(f"Questions answered: {len(results)}/{total}\n")
            f.write("=" * 70 + "\n\n")
            for r in results:
                f.write(f"Q{r['q']:03d}. {r['question']}\n")
                f.write("-" * 70 + "\n")
                f.write(r["answer"] + "\n\n")

    print(f"\n{'='*70}")
    print(f"  Done! {len(results)} questions answered.")
    print(f"  Readable results : {txt_out}")
    print(f"  JSON results     : {json_out}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
