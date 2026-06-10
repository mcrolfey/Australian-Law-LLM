# 🇦🇺 Australian Law LLM

Fine-tune a LLaMA model on the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) locally using [Unsloth](https://github.com/unslothai/unsloth), then serve it as a **localhost chat UI**.

The corpus covers legislation, case law, and legal instruments across all Australian jurisdictions (Commonwealth, NSW, VIC, QLD, WA, SA, TAS, ACT, NT).

---

## Features

- **4-bit quantised training** — runs on consumer GPUs from 4 GB VRAM upward
- **GPU tier configs** — one flag selects the right model and batch settings for your card
- **LoRA fine-tuning** via Unsloth (2–5× faster than vanilla HuggingFace)
- **Localhost Gradio chat UI** — no cloud, no API keys, fully offline after setup
- **Windows compatible** — triton ordering, pickle, and multiprocessing issues all patched

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| GPU VRAM  | 4 GB minimum (GTX 1650 / RTX 3050) |
| RAM       | 16 GB |
| Disk      | ~20 GB (model + dataset) |
| OS        | Windows 10/11, Ubuntu 20.04+ |
| Python    | **3.10, 3.11, or 3.12 only** — 3.13 is not supported by PyTorch CUDA |
| CUDA      | 12.1 or 12.4 |

> ⚠️ **Python 3.13 will not work.** PyTorch CUDA wheels only exist for 3.10–3.12.
> Install Python 3.11 via `winget install Python.Python.3.11` then create your venv with `py -3.11 -m venv venv`.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/mcrolfey/Australian-Law-LLM.git
cd Australian-Law-LLM
```

### 2. Create a virtual environment with Python 3.11

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Linux / macOS
python3.11 -m venv venv
source venv/bin/activate

# Verify
python --version   # must show 3.11.x
```

### 3. Install PyTorch with CUDA

**This step must be done before installing anything else.**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify your GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True / NVIDIA GeForce RTX ...
```

### 4. Install triton (Windows only)

On Windows, triton must be installed at version **3.2.x**. Newer versions (3.7+) break `torch._inductor` and cause a silent crash.

```bash
pip install triton-windows==3.2.0.post21
```

> Linux users: skip this step — triton is bundled with PyTorch on Linux.

### 5. Install Unsloth and remaining dependencies

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers==4.56.1
pip install -r requirements.txt
```

> **Why `transformers==4.56.1`?** Unsloth 2026.x requires transformers in the range `4.51.3–5.5.0` but blocks several versions (4.52–4.55, 5.0, 5.1). Version 4.56.1 is the stable target.

### 6. Set up Kaggle credentials

The dataset is downloaded automatically via [kagglehub](https://github.com/Kaggle/kagglehub). You need a free Kaggle account:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Place the downloaded `kaggle.json` at:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

---

## Training

### Pick your GPU tier

```bash
python train.py --list-configs
```

Output:
```
Available GPU configs:

  --gpu 4gb    GTX 1650 / RTX 3050 / low VRAM   → Llama-3.2-1B,  seq=256
  --gpu 8gb    RTX 3070 / RTX 3060 / mid VRAM   → Llama-3.2-3B,  seq=512
  --gpu 16gb   RTX 4080 / RTX 3080 Ti / hi VRAM → Llama-3.2-3B,  seq=1024
  --gpu 24gb   RTX 3090 / RTX 4090 / max VRAM   → Llama-3.1-8B,  seq=2048
```

> GPU auto-detection requires CUDA-enabled PyTorch. If it shows `0.0 GB`, your CUDA install isn't visible — run the verify command from step 3.

### Run training

```bash
python train.py --gpu 8gb

# Override number of steps
python train.py --gpu 16gb --steps 200
```

Training saves LoRA adapters to `lora_australian_law_model/` when complete (~4–5 min for 60 steps on an RTX 3070).

### Config summary

| Flag | Model | Seq Len | Batch | Steps | Est. VRAM |
|------|-------|---------|-------|-------|-----------|
| `--gpu 4gb`  | Llama-3.2-1B | 256  | 1×16 | 60  | ~3.5 GB |
| `--gpu 8gb`  | Llama-3.2-3B | 512  | 1×8  | 60  | ~6.5 GB |
| `--gpu 16gb` | Llama-3.2-3B | 1024 | 2×4  | 120 | ~12 GB  |
| `--gpu 24gb` | Llama-3.1-8B | 2048 | 2×4  | 200 | ~20 GB  |

---

## Serving the Web UI

After training, launch the Gradio chat interface:

```bash
python serve.py
```

Opens **http://localhost:7860** automatically.

### Options

```bash
# Specify a different adapter directory
python serve.py --model lora_australian_law_model

# Change port
python serve.py --port 8080

# Load a fully merged (non-LoRA) model
python serve.py --model merged_model --merged
```

### Web UI features

- Multi-turn chat with conversation history
- Adjustable max token generation slider
- Example prompts to get started
- Australian legal system-prompt (jurisdiction, citation, disclaimer-aware)

---

## Project Structure

```
Australian-Law-LLM/
├── train.py              # Main training entry point
├── serve.py              # Gradio web UI server
├── requirements.txt
├── configs/
│   ├── gpu_4gb.py        # 4 GB VRAM settings
│   ├── gpu_8gb.py        # 8 GB VRAM settings
│   ├── gpu_16gb.py       # 16 GB VRAM settings
│   └── gpu_24gb.py       # 24 GB VRAM settings
└── lora_australian_law_model/   # Generated after training (gitignored)
```

---

## Dataset

**[Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus)** by Umar Butler.

The largest open database of Australian law, containing:
- Acts and regulations (Commonwealth + all states/territories)
- Federal and state court judgments
- Tribunal decisions

Each document is formatted using the **Alpaca instruction template**:

```
### Instruction:
Analyze the following Australian legal document...

### Input:
Citation: Mabo v Queensland (No 2) [1992] HCA 23
Jurisdiction: Commonwealth
Document Type: decision

### Response:
<document text>
```

---

## Troubleshooting

**Out of memory (OOM)**
- Switch to the next lower GPU tier: `--gpu 4gb`
- Reduce steps: `--steps 30`
- Close other GPU-intensive applications before running

**Silent crash / no output after Unsloth banner**
- Caused by `triton-windows` version mismatch. Reinstall: `pip install triton-windows==3.2.0.post21`
- Also check for a stale `torchao` install: `pip uninstall torchao -y`

**`transformers` version conflict**
- Pin to the known-good version: `pip install transformers==4.56.1`

**`kagglehub` authentication error**
- Verify `kaggle.json` is in the correct location and has `600` permissions (Linux) or is readable (Windows)

**Windows pickle / multiprocessing errors**
- The trainer uses `dataset_num_proc=1` and `save_strategy="no"` to prevent this — already configured.

**Gradio port already in use**
- `python serve.py --port 7861`

**GPU not detected by `--list-configs`**
- Run `python -c "import torch; print(torch.cuda.get_device_name(0))"` to verify CUDA is working
- If that fails, reinstall PyTorch CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

---

## Disclaimer

This project is for **research and educational purposes only**. The fine-tuned model does not constitute legal advice. Always consult a qualified Australian lawyer for legal matters.

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — fast fine-tuning with minimal VRAM
- [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) — Umar Butler
- [Meta LLaMA](https://llama.meta.com/) — base model
