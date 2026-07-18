"""
Australian Law LLM - Q&A Pair Generator
========================================
Samples documents from the Open Australian Legal Corpus and uses a local
LM Studio model to generate question/answer pairs for SFT training.

Output: a JSONL file where each line is:
  {"question": "...", "answer": "...", "source": "citation", "jurisdiction": "..."}

Usage:
    python generate_qa.py --count 5000 --pairs-per-doc 3
    python generate_qa.py --count 10000 --output qa_dataset.jsonl --resume
    python generate_qa.py --count 1000 --lm-studio-url http://localhost:1234
"""

import argparse
import json
import os
import random
import re
import sys
import time

KAGGLE_CACHE = os.path.expanduser(
    "~/.cache/kagglehub/datasets/umarbutler/open-australian-legal-corpus/versions/2/corpus.jsonl"
)

# Prompt sent to LM Studio for each document chunk.
# Instructs it to return ONLY a JSON array — no prose.
QA_GENERATION_PROMPT = """\
You are a legal question-answer pair generator for Australian law.

Given the following excerpt from an Australian legal document, generate {n} \
question-answer pairs that test comprehension of the legal content.

Rules:
- Questions must be answerable solely from the provided text.
- Answers must be accurate, concise, and directly grounded in the text.
- Do NOT invent facts not present in the text.
- Output ONLY a JSON array, no other text.

Format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

LEGAL TEXT:
{text}

JSON OUTPUT:
"""


def parse_args():
    p = argparse.ArgumentParser(description="Generate Q&A pairs from the legal corpus")
    p.add_argument("--count", type=int, default=5000,
                   help="Number of documents to process (default: 5000)")
    p.add_argument("--pairs-per-doc", type=int, default=3,
                   help="Q&A pairs to request per document (default: 3)")
    p.add_argument("--max-doc-tokens", type=int, default=800,
                   help="Max words per document chunk sent to LM Studio (default: 800)")
    p.add_argument("--output", default="qa_dataset.jsonl",
                   help="Output JSONL file (default: qa_dataset.jsonl)")
    p.add_argument("--lm-studio-url", default="http://localhost:1234",
                   help="LM Studio base URL (default: http://localhost:1234)")
    p.add_argument("--lm-studio-model", default=None,
                   help="Model ID in LM Studio (auto-detected if omitted)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for document sampling (default: 42)")
    p.add_argument("--resume", action="store_true",
                   help="Skip documents already in the output file and continue")
    p.add_argument("--corpus", default=None,
                   help="Path to corpus.jsonl (auto-detected from Kaggle cache if omitted)")
    p.add_argument("--debug", action="store_true",
                   help="Print raw LM Studio response whenever parsing fails")
    return p.parse_args()


def get_lm_client(base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.  Run: pip install openai")
        sys.exit(1)
    return OpenAI(base_url=f"{base_url}/v1", api_key="lm-studio")


def detect_model(client) -> str:
    try:
        models = client.models.list()
        model_id = models.data[0].id
        print(f"  Auto-detected LM Studio model: {model_id}")
        return model_id
    except Exception as e:
        print(f"ERROR: Cannot reach LM Studio — {e}")
        sys.exit(1)


def load_corpus(path: str) -> list[dict]:
    print(f"Loading corpus: {path}")
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    print(f"  {len(docs):,} documents loaded")
    return docs


def truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _extract_from_400_body(err) -> str | None:
    """
    Gemma 4 thinking models cause LM Studio to return a 400 error whose body
    contains the full thinking output (including the Q&A pairs we want).
    Try to pull that text out so parse_qa_response can still work on it.
    """
    try:
        body = getattr(err, "body", None)
        if body and isinstance(body, dict):
            msg = body.get("error", "")
        else:
            msg = str(err)
        # The error message is: "Failed to parse input at pos 0: <thinking content>"
        marker = "Failed to parse input at pos 0: "
        idx = msg.find(marker)
        if idx != -1:
            return msg[idx + len(marker):]
        return msg if msg else None
    except Exception:
        return None


def call_lm_studio(client, model: str, prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status == 400:
                # Gemma 4 is a thinking model — LM Studio returns 400 when the
                # output contains <|channel>thought\n...> reasoning blocks, but
                # the error body contains the full output including the Q&A pairs.
                extracted = _extract_from_400_body(e)
                if extracted:
                    return extracted
            wait = 2 ** attempt
            print(f"    LM Studio error (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    return None


def _valid_pair(p) -> bool:
    return (
        isinstance(p, dict)
        and "question" in p and "answer" in p
        and isinstance(p["question"], str) and p["question"].strip()
        and isinstance(p["answer"],   str) and p["answer"].strip()
    )


def parse_qa_response(response: str) -> list[dict]:
    """
    Robustly extract Q&A pairs from an LM Studio response.

    Handles all common Gemma output patterns:
      • Clean JSON array:           [{"question":...}, ...]
      • Markdown fenced:            ```json\n[...]\n```
      • Array inside prose:         "Here are pairs:\n[...]"
      • Newline-separated objects:  {"question":...}\n{"question":...}
    """
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```[a-z]*\n?", "", response).strip()
    text = re.sub(r"\n?```", "", text).strip()

    # Strategy 1: parse as a JSON array
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1:
        try:
            pairs = json.loads(text[start:end+1])
            if isinstance(pairs, list):
                valid = [p for p in pairs if _valid_pair(p)]
                if valid:
                    return valid
        except json.JSONDecodeError:
            pass

    # Strategy 2: find all individual JSON objects and collect valid pairs
    pairs = []
    for match in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if _valid_pair(obj):
                pairs.append(obj)
        except json.JSONDecodeError:
            pass
    if pairs:
        return pairs

    # Strategy 3: model used "Q:" / "A:" / "Question:" / "Answer:" plain-text format.
    # Extract all questions and answers separately, then zip them.
    # This is more robust than a single combined regex when there are decorators
    # (bullet points, "Idea N:" headers, etc.) between pairs.
    questions = re.findall(
        r"[Qq](?:uestion)?[:\.\)]\s*(.+?)(?=\s*[Aa](?:nswer)?[:\.\)])",
        text, re.DOTALL
    )
    answers = re.findall(
        r"[Aa](?:nswer)?[:\.\)]\s*(.+?)(?=\s*[Qq](?:uestion)?[:\.\)]|$)",
        text, re.DOTALL
    )
    if questions and answers:
        pairs = [
            {"question": q.strip(), "answer": a.strip()}
            for q, a in zip(questions, answers)
            if q.strip() and a.strip()
        ]
        if pairs:
            return pairs

    return []


def count_existing(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    count = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    args = parse_args()

    # Locate corpus
    corpus_path = args.corpus or KAGGLE_CACHE
    if not os.path.exists(corpus_path):
        print(f"ERROR: corpus not found at {corpus_path}")
        print("Run the main training pipeline once to download it, or pass --corpus <path>")
        sys.exit(1)

    # LM Studio
    print("Connecting to LM Studio...")
    client = get_lm_client(args.lm_studio_url)
    model  = args.lm_studio_model or detect_model(client)

    # Load and sample corpus
    docs = load_corpus(corpus_path)
    random.seed(args.seed)
    sample = random.sample(docs, min(args.count, len(docs)))
    print(f"  Sampled {len(sample):,} documents\n")

    # Resume support
    already_done = 0
    if args.resume:
        already_done = count_existing(args.output)
        print(f"  Resuming — {already_done:,} pairs already in {args.output}")

    # Open output file (append if resuming)
    mode = "a" if args.resume else "w"
    total_pairs = already_done
    errors = 0

    # How many docs to skip if resuming (rough estimate: pairs_per_doc pairs per doc)
    docs_to_skip = already_done // args.pairs_per_doc if args.resume else 0

    print(f"Generating Q&A pairs ({args.pairs_per_doc} per document)...\n")
    t_start = time.time()

    with open(args.output, mode, encoding="utf-8") as out_f:
        for i, doc in enumerate(sample):
            if i < docs_to_skip:
                continue

            text = (doc.get("text") or "").strip()
            if len(text.split()) < 50:
                continue  # too short to generate useful Q&A

            citation    = doc.get("citation", "Unknown")
            jurisdiction = doc.get("jurisdiction", "Unknown")
            doc_type    = doc.get("type", "Unknown")

            chunk = truncate_to_words(text, args.max_doc_tokens)
            prompt = QA_GENERATION_PROMPT.format(n=args.pairs_per_doc, text=chunk)

            response = call_lm_studio(client, model, prompt)
            if response is None:
                errors += 1
                continue

            pairs = parse_qa_response(response)
            if not pairs:
                errors += 1
                if errors <= 5:
                    print(f"  [doc {i+1}] Failed to parse response — skipping")
                    if args.debug:
                        print(f"  RAW RESPONSE:\n{response[:600]}\n  ---")
                continue

            for pair in pairs:
                record = {
                    "question":     pair["question"].strip(),
                    "answer":       pair["answer"].strip(),
                    "source":       citation,
                    "jurisdiction": jurisdiction,
                    "type":         doc_type,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += 1

            out_f.flush()

            # Progress every 50 docs
            if (i + 1) % 50 == 0:
                elapsed = (time.time() - t_start) / 60
                rate    = (i + 1 - docs_to_skip) / elapsed if elapsed > 0 else 0
                remaining = (len(sample) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>5}/{len(sample)}]  {total_pairs:,} pairs written  "
                      f"({elapsed:.1f} min elapsed, ~{remaining:.0f} min remaining)")

    elapsed_total = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  Done! {total_pairs:,} Q&A pairs written to {args.output}")
    print(f"  {errors} documents skipped (parse errors or LM Studio failures)")
    print(f"  Total time: {elapsed_total:.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
