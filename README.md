# Australian Law LLM

Fine-tune a LLaMA model on the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) entirely on your local machine, then chat with it through a browser. No cloud. No API keys. No internet required after the initial download.

The corpus covers 202,000+ documents — legislation, case law, and legal instruments — across all Australian jurisdictions: Commonwealth, NSW, VIC, QLD, WA, SA, TAS, ACT, and NT.

---

## Features

- **Two-step training pipeline** — Continued Pre-Training (CPT) followed by instruction SFT reduces hallucination by grounding the model in actual legal text before teaching it to answer questions
- **4-bit quantised training** — runs on consumer GPUs from 4 GB VRAM upward
- **GPU tier configs** — one flag selects the right model size and batch settings for your hardware
- **LoRA fine-tuning via Unsloth** — 2–5× faster than vanilla HuggingFace with lower memory usage
- **Localhost Gradio chat UI** — open your browser and talk to your model
- **Batch evaluation** — automatically run 100 benchmark legal questions and compare fine-tuned vs base model answers
- **RAG open-book evaluation** — supply retrieved legal text at inference time; the model answers strictly from context, eliminating parametric hallucination
- **Windows compatible** — all known Windows issues patched (triton import order, pickle errors, multiprocessing, pyarrow DLL crash, venv PATH)

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| GPU VRAM  | 4 GB minimum (GTX 1650 / RTX 3050 or better) |
| RAM       | 16 GB |
| Disk      | ~25 GB (models + dataset + outputs) |
| OS        | Windows 10/11 or Ubuntu 20.04+ |
| Python    | **3.10, 3.11, or 3.12** — Python 3.13 has no PyTorch CUDA wheels |
| CUDA      | 12.1 or 12.4 |

> **Windows note:** This repo is developed and tested on Windows with an RTX 3070 Laptop (8 GB VRAM). All Windows-specific workarounds are already applied in the scripts.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/mcrolfey/Australian-Law-LLM.git
cd Australian-Law-LLM
```

### 2. Create a conda environment

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is strongly recommended on Windows. It handles PATH correctly and avoids the activation issues that plague `venv` when Miniconda is also installed system-wide.

```bash
conda create -n aus-law python=3.11 -y
conda activate aus-law

# Confirm the right Python is active
python --version   # must print 3.11.x
```

> **Linux / macOS alternative:** `python3.11 -m venv venv && source venv/bin/activate`

### 3. Install PyTorch with CUDA

This must happen before anything else. PyTorch's CUDA build is not available through the default pip index.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify your GPU is visible before continuing:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX ...
```

If this prints `False`, stop here and fix your CUDA installation before proceeding.

### 4. Install triton (Windows only)

On Windows, triton must be pinned to **3.2.x**. Version 3.7+ breaks `torch._inductor` and causes Unsloth to crash silently with no error message.

```bash
pip install triton-windows==3.2.0.post21
```

> Linux users: skip this step — triton is included with PyTorch on Linux.

### 5. Install Unsloth

Unsloth must be installed from GitHub, not PyPI. The PyPI version is not kept up to date.

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 6. Install remaining dependencies

```bash
pip install transformers==4.56.1
pip install -r requirements.txt
```

The `transformers` pin comes before `requirements.txt` to prevent pip from upgrading it. Unsloth 2026.x requires transformers in a specific range and several versions in that range are blocked — 4.56.1 is the confirmed working version.

### 7. Set up Kaggle credentials

The Open Australian Legal Corpus is downloaded via [kagglehub](https://github.com/Kaggle/kagglehub). You need a free Kaggle account.

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Move the downloaded `kaggle.json` to:
   - **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`

The dataset is downloaded on first run and cached locally. Subsequent runs use the cache and do not need internet access.

---

## Training Pipeline

The full pipeline has three stages. Each stage builds on the last.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 — CPT (cpt_train.py)                                       │
│  Feed raw Acts and judgments. Model learns what the law says.        │
│  Output: lora_cpt_law_model/                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2 — SFT (train.py)                                           │
│  Teach the model to answer questions in Alpaca format.               │
│  Optionally starts from CPT weights (--cpt-model flag).              │
│  Output: lora_australian_law_model/                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3 — Self-Evolution (self_evolve.py)                           │
│  Trains for N steps → evaluates all 100 benchmark questions →        │
│  sends loss curve + answers to LM Studio (Gemma 4 26B locally) →    │
│  LM Studio suggests hyperparameter changes → repeats for K rounds.  │
│  Output: self_evolve_output/round_NN_adapter/ + round_NN_answers.txt │
└─────────────────────────────────────────────────────────────────────┘
```

You can run any stage independently. Running all three gives the best results: CPT grounds the model in real legal text, SFT teaches it to answer questions, and self-evolution iteratively improves the hyperparameters using a local LLM as the judge.

### Why three stages?

Training directly on Q&A pairs (SFT alone) causes hallucination — the model has no knowledge of specific Acts, sections, or cases, so it generates confident-sounding but incorrect answers. CPT fixes the factual gap. Self-evolution then continuously improves training quality by observing the model's actual answers across 100 benchmark questions, not just the loss curve.

---

### Step 1: Continued Pre-Training (CPT)

CPT feeds raw legal text to the model with no instruction template — just dense, packed sequences of Acts and judgments. This injects factual knowledge directly into the model's weights.

```bash
# Using the Open Australian Legal Corpus (default)
python cpt_train.py --gpu 8gb

# Using your own local documents
python cpt_train.py --gpu 8gb --data-dir ./law_texts

# List available GPU configs
python cpt_train.py --list-configs
```

**CPT GPU configs:**

| Flag | Model | Seq Len | LoRA Rank | Steps | Est. Time (RTX 3070) |
|------|-------|---------|-----------|-------|----------------------|
| `--gpu 4gb`  | Llama-3.2-1B | 2048 | 32 | 200 | ~25 min |
| `--gpu 8gb`  | Llama-3.2-1B | 2048 | 32 | 300 | ~40 min |
| `--gpu 16gb` | Llama-3.2-3B | 8192 | 64 | 500 | ~3 hr   |
| `--gpu 24gb` | Llama-3.1-8B | 8192 | 64 | 500 | ~4 hr   |

> **8 GB laptop GPU note:** Laptop GPUs share VRAM with the display driver, leaving less than the card's rated capacity for training. The `--gpu 8gb` CPT config uses the 1B model to stay within this constraint. If you have a desktop GPU with the display on a separate card, you can try `--gpu 16gb`.

**Input data formats for `--data-dir`:**

*JSONL (recommended)* — one document per line with a `text` key:
```json
{"text": "Corporations Act 2001 (Cth)\n\nPart 1.1 — Preliminary\n\n1 Short title..."}
{"text": "Mabo v Queensland (No 2) [1992] HCA 23\n\nBrennan J: The common law..."}
```

*Plain text* — one `.txt` file per document in a folder:
```
law_texts/
  corporations_act_2001.txt
  mabo_v_queensland_1992.txt
```

---

### Step 2: SFT Instruction Fine-Tuning

SFT teaches the model to respond to questions in a structured, helpful way using the Alpaca instruction format.

```bash
# List available GPU configs
python train.py --list-configs

# Train (standard)
python train.py --gpu 8gb

# Train with more steps
python train.py --gpu 8gb --steps 200
```

**SFT GPU configs:**

| Flag | Model | Seq Len | Steps | LR | Est. Time (RTX 3070) |
|------|-------|---------|-------|----|----------------------|
| `--gpu 4gb`  | Llama-3.2-1B | 256  | 1000 | 5e-5 | ~60 min  |
| `--gpu 8gb`  | Llama-3.2-3B | 512  | 1000 | 5e-5 | ~75 min  |
| `--gpu 16gb` | Llama-3.2-3B | 1024 | 1000 | 5e-5 | ~3 hr    |
| `--gpu 24gb` | Llama-3.1-8B | 2048 | 1000 | 5e-5 | ~5 hr    |

All configs use cosine LR decay, 50 warmup steps, weight decay 0.1. Override steps with `--steps N` for a quicker test run.

Training saves LoRA adapters to `lora_australian_law_model/` on completion.

---

### Stage 3: Recursive Self-Evolution

Self-evolution runs a closed loop: train → evaluate → consult LM Studio → update hyperparameters → repeat. Unlike standard training, it observes the model's actual answers — not just the loss — and uses a local LLM (running in LM Studio) to decide how to improve.

**Prerequisites:**
1. LM Studio installed and running with a model loaded (tested with `google/gemma-4-26b-a4b`)
2. LM Studio local server enabled (default port 1234)
3. `pip install openai` (uses the OpenAI-compatible LM Studio API)

```bash
# 5 rounds of 200 steps each — recommended starting point
python self_evolve.py --gpu 8gb --rounds 5 --steps-per-round 200

# More rounds for deeper optimisation
python self_evolve.py --gpu 8gb --rounds 10 --steps-per-round 100

# Custom LM Studio URL
python self_evolve.py --gpu 8gb --rounds 5 --steps-per-round 200 --lm-studio-url http://localhost:1234
```

**What happens each round:**

1. **Train** — fine-tune for `--steps-per-round` steps with the current hyperparameters
2. **Evaluate** — run all 100 benchmark questions through the freshly trained adapter and save answers to `round_NN_answers.txt`
3. **Consult** — send the loss curve and all 100 answers to LM Studio; Gemma 4's 128k context window means every answer is sent in full with no truncation
4. **Update** — apply the hyperparameter changes suggested by LM Studio (learning rate, LoRA rank, batch size, weight decay, warmup steps), clamped to safe bounds
5. **Repeat** for the next round

**Output files per round:**

| File | Contents |
|------|----------|
| `self_evolve_output/round_NN_adapter/` | LoRA adapter weights for that round |
| `self_evolve_output/round_NN_answers.txt` | All 100 benchmark answers — readable |
| `self_evolve_output/round_NN_answers.json` | All 100 benchmark answers — structured |
| `self_evolve_output/evolution_log.json` | Full history: configs, loss curves, LM Studio responses, hyperparameter changes |

At the end, the script identifies the round with the lowest final loss and prints its adapter path. Compare `round_01_answers.txt` with `round_05_answers.txt` to see how legal reasoning improved.

**Hyperparameter bounds** (LM Studio cannot push values outside these):

| Parameter | Range |
|-----------|-------|
| `learning_rate` | 1e-6 — 1e-3 |
| `per_device_train_batch_size` | 1 — 4 |
| `gradient_accumulation_steps` | 1 — 32 |
| `warmup_steps` | 0 — 200 |
| `weight_decay` | 0.0 — 0.3 |
| `r` (LoRA rank) | 4 — 64 |
| `lora_alpha` | 4 — 128 |

---

## Chat Interface

After training, launch the web UI:

```bash
python serve.py
```

This opens **http://localhost:7860** in your browser automatically.

```bash
# Options
python serve.py --model lora_australian_law_model   # explicit model path
python serve.py --port 8080                          # different port
python serve.py --model merged_model --merged        # fully merged model
```

The chat UI supports multi-turn conversation, adjustable generation length, and example prompts covering common areas of Australian law.

---

## Batch Evaluation

Run 100 benchmark legal questions through the model automatically and save every answer to a file. Covers constitutional law, criminal law, contract, tort, corporations, property, family law, administrative law, IP, employment, and procedure.

```bash
# cd to the project folder first so output files land here
cd C:\Users\<you>\Australian-Law-LLM

# Test the fine-tuned model
python batch_test.py

# Test the base model (no fine-tuning) for comparison
python batch_test.py --base-only

# Resume a run that was interrupted at question 42
python batch_test.py --start 42
```

Results are written after every question so no progress is lost if the run is interrupted.

**Output files:**

| File | Contents |
|------|----------|
| `batch_results_finetuned.txt` | Readable Q&A results — fine-tuned model |
| `batch_results_base.txt` | Readable Q&A results — base model |
| `batch_results_finetuned.json` | Structured JSON — fine-tuned model |
| `batch_results_base.json` | Structured JSON — base model |

**All options:**

```
--model PATH        LoRA adapter directory (default: lora_australian_law_model)
--base-only         Load the base model only, without LoRA adapters
--base-model NAME   Override the base model name or path
--max-tokens N      Maximum tokens per answer (default: 512)
--start N           Resume from question N (1-indexed)
--output FILE       Override the .txt output filename
--json-output FILE  Override the .json output filename
```

At ~8 s/question on an RTX 3070, all 100 questions take roughly 15 minutes.

---

## Training on Google Colab (larger models)

Your local GPU limits you to the 3B model. To train the 8B model, use the included Colab notebook — it auto-detects the GPU, installs everything in the right order, runs CPT + SFT, and downloads the adapter files back to your machine.

Open [`colab_train.ipynb`](colab_train.ipynb) in [Google Colab](https://colab.research.google.com/) and set the runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`).

| Colab runtime | VRAM | Model trained | CPT time | SFT time |
|---------------|------|---------------|----------|----------|
| T4 (free)     | 16 GB | Llama-3.2-3B | ~2 hr    | ~20 min  |
| L4 (Pro+)     | 24 GB | Llama-3.1-8B | ~2 hr    | ~30 min  |
| A100 (Pro)    | 40 GB | Llama-3.1-8B | ~45 min  | ~15 min  |

The notebook downloads the finished adapters as a zip file. Unzip into your local `Australian-Law-LLM/` folder and serve with:

```bash
python serve.py --model lora_australian_law_model
```

The notebook also has an optional cell to push adapters to a private HuggingFace repo, so you can pull them on any machine without re-training.

---

## Recursive Self-Evolution (LM Studio)

`self_evolve.py` runs a training loop where a local LLM (via LM Studio) analyses the loss curve after each round and rewrites the hyperparameters for the next round — no cloud, no manual tuning.

### How it works

```
Round 1: train N steps → capture loss curve
         ↓
         Send loss curve + current config to LM Studio
         LM Studio returns a JSON config patch (new LR, rank, etc.)
         Apply patch (clamped to safe bounds)
         ↓
Round 2: train N steps with updated config → repeat
```

The LLM sees the last 3 rounds of history so it can track whether its changes are actually helping.

### Prerequisites

1. [LM Studio](https://lmstudio.ai/) installed and running with any model loaded
2. Local server enabled in LM Studio (`☰ → Local Server → Start Server`)
3. `pip install openai`

### Run

```bash
# 5 rounds, 200 steps each (recommended first run — ~2 hr total on RTX 3070)
python self_evolve.py --gpu 8gb --rounds 5 --steps-per-round 200

# Quick test — 3 rounds, 50 steps each
python self_evolve.py --gpu 8gb --rounds 3 --steps-per-round 50

# Custom LM Studio URL or model
python self_evolve.py --gpu 8gb --rounds 5 --steps-per-round 200 \
    --lm-studio-url http://localhost:1234 \
    --lm-studio-model "llama-3-8b-instruct"
```

### Outputs

All outputs land in `self_evolve_output/` (override with `--output-dir`):

| Path | Contents |
|------|----------|
| `self_evolve_output/evolution_log.json` | Full history: every round's config, loss curve, LM Studio response, and applied changes |
| `self_evolve_output/round_01_adapter/` | LoRA adapter after round 1 |
| `self_evolve_output/round_02_adapter/` | LoRA adapter after round 2 (with updated config) |
| … | … |

After the loop finishes, the script reports which round achieved the lowest end-of-round loss. Serve that adapter directly:

```bash
python serve.py --model self_evolve_output/round_04_adapter
```

### Tunable parameters

LM Studio can only modify parameters within these safe bounds — values outside are clamped automatically:

| Parameter | Range |
|-----------|-------|
| `learning_rate` | 1e-6 – 1e-3 |
| `per_device_train_batch_size` | 1 – 4 |
| `gradient_accumulation_steps` | 1 – 32 |
| `warmup_steps` | 0 – 200 |
| `weight_decay` | 0.0 – 0.3 |
| `r` (LoRA rank) | 4 – 64 |
| `lora_alpha` | 4 – 128 |

Model architecture, sequence length, and dataset are not modified — only the optimisation hyperparameters.

---

## Interactive Testing

Test a single question from the command line without launching the web UI:

```bash
python test_model.py "What are the elements of negligence under Australian common law?"

# Interactive mode (question/answer loop)
python test_model.py
```

---

## RAG / Open-Book Evaluation

If the fine-tuned model hallucinates — inventing case names, section numbers, or statutory text that doesn't exist — the RAG evaluation script forces it to work only from retrieved legal text supplied at inference time.

This tests a different capability: given the actual words of an Act or judgment, can the model extract and explain the relevant rule? It is a more reliable evaluation of legal reasoning than the closed-book batch test.

### Input format

Create a JSON file with `question` and `context` fields. A ready-made `test_data.json` with 30 questions is included:

```json
[
  {
    "question": "What are the elements of murder under the Crimes Act 1900 (NSW)?",
    "context": "Section 18 of the Crimes Act 1900 (NSW) provides: '(1)(a) Murder shall be taken to have been committed where the act of the accused..."
  }
]
```

CSV is also accepted (columns named `question` and `context`).

### Run

```bash
# Open-book eval using the fine-tuned model
python rag_eval.py --input test_data.json

# Compare against the base model (no LoRA)
python rag_eval.py --input test_data.json --base-only

# Resume an interrupted run from row 15
python rag_eval.py --input test_data.json --start 15

# Longer answers
python rag_eval.py --input test_data.json --max-tokens 800
```

**Output files:**

| File | Contents |
|------|----------|
| `rag_batch_results.txt` | Readable context-grounded answers |
| `rag_batch_results.json` | Structured JSON with question, context length, and answer |

**All options:**

```
--input FILE        Path to .json or .csv test file (required)
--model PATH        LoRA adapter directory (default: lora_australian_law_model)
--base-only         Load the base model only, without LoRA adapters
--base-model NAME   Override the base model name or path
--max-tokens N      Maximum tokens per answer (default: 512)
--start N           Resume from row N (1-indexed)
--output FILE       Override the .txt output filename
--json-output FILE  Override the .json output filename
```

The prompt template uses hard constraints instructing the model not to use parametric memory, not to fabricate information, and to explicitly state when the answer is not in the provided context.

---

## Project Structure

```
Australian-Law-LLM/
├── cpt_train.py               # Step 1: CPT on raw legal text
├── train.py                   # Step 2: SFT instruction fine-tuning
├── colab_train.ipynb          # Colab notebook — train 8B on a free T4 GPU
├── serve.py                   # Gradio localhost chat UI
├── batch_test.py              # Batch eval: 100 questions, fine-tuned vs base
├── rag_eval.py                # RAG open-book eval: context-grounded answers
├── test_data.json             # 30 question/context pairs for rag_eval.py
├── self_evolve.py             # Recursive self-evolution loop via LM Studio
├── test_model.py              # Interactive single-question CLI
├── requirements.txt
├── configs/
│   ├── gpu_4gb.py             # SFT settings for 4 GB VRAM
│   ├── gpu_8gb.py             # SFT settings for 8 GB VRAM
│   ├── gpu_16gb.py            # SFT settings for 16 GB VRAM
│   └── gpu_24gb.py            # SFT settings for 24 GB VRAM
├── lora_cpt_law_model/        # CPT output — created by cpt_train.py (gitignored)
└── lora_australian_law_model/ # SFT output — created by train.py (gitignored)
```

---

## Troubleshooting

**Silent crash immediately after the Unsloth banner**
The most common cause on Windows is a triton version mismatch or a pyarrow DLL access violation. Fix in order:
```bash
pip install triton-windows==3.2.0.post21
pip install pyarrow==17.0.0 datasets==3.5.0
pip install torchao==0.13.0
```
To diagnose which component is crashing:
```bash
python -X faulthandler -c "import triton; import torch; from unsloth import FastLanguageModel; print('OK')"
```

**`pyarrow` access violation / crash on import**
pyarrow 21+ has a DLL incompatibility on some Windows configurations. Pin to the known-good version:
```bash
pip install pyarrow==17.0.0 datasets==3.5.0
```

**Out of memory (OOM) during training**
- Use a lower GPU tier: `--gpu 4gb`
- Reduce steps to test first: `--steps 10`
- Close other GPU-intensive applications (games, other ML jobs)
- On a laptop, the display driver takes VRAM — this is accounted for in the configs

**Model download stalls at 0 B/s**
The scripts retry automatically (up to 20 times, 60-second socket timeout). Install `hf_transfer` for faster and more reliable downloads:
```bash
pip install hf_transfer
```

**`transformers` version conflict**
```bash
pip install transformers==4.56.1
```

**`unsloth_zoo` errors**
```bash
pip install "unsloth_zoo==2026.6.1"
```

**Kaggle authentication error**
Confirm `kaggle.json` exists at `C:\Users\<you>\.kaggle\kaggle.json` and contains valid credentials. After the first successful download the corpus is cached locally and Kaggle credentials are no longer needed.

**GPU not detected**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If this fails, reinstall PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Gradio port already in use**
```bash
python serve.py --port 7861
```

**Windows multiprocessing / pickle errors during training**
Already handled — all scripts use `dataset_num_proc=1` and `save_strategy="no"`.

---

## Dataset

**[Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus)** by Umar Butler — the largest open database of Australian law, with 202,000+ documents covering:

- Commonwealth, state, and territory Acts and regulations
- Federal, state, and territory court judgments
- Tribunal decisions

---

## Disclaimer

This project is for **research and educational purposes only**. The fine-tuned model does not provide legal advice and should not be relied upon for legal decisions. Always consult a qualified Australian lawyer.

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA fine-tuning
- [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) — Umar Butler
- [Meta LLaMA](https://llama.meta.com/) — base model family
