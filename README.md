# 🇦🇺 Australian Law LLM

Fine-tune a LLaMA model on the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) locally using [Unsloth](https://github.com/unslothai/unsloth), then serve it as a **localhost chat UI**.

The corpus covers legislation, case law, and legal instruments across all Australian jurisdictions (Commonwealth, NSW, VIC, QLD, WA, SA, TAS, ACT, NT).

---

## Features

- **4-bit quantised training** — runs on consumer GPUs from 4 GB VRAM upward
- **GPU tier configs** — pick the right settings for your hardware automatically
- **LoRA fine-tuning** via Unsloth (2–5× faster than vanilla HuggingFace)
- **Localhost Gradio chat UI** — no cloud, no API keys, fully offline after setup
- **Windows compatible** — pickle/multiprocessing issues patched

---

## Requirements

| Component | Minimum |
|-----------|---------|
| GPU VRAM  | 4 GB (GTX 1650 / RTX 3050) |
| RAM       | 16 GB |
| Disk      | ~20 GB (model + dataset) |
| OS        | Windows 10/11, Ubuntu 20.04+, macOS (CPU only) |
| Python    | 3.10 or 3.11 |
| CUDA      | 11.8 or 12.1 |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/mcrolfey/Australian-Law-LLM.git
cd Australian-Law-LLM
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Unsloth note:** If the above fails, install Unsloth separately first following the [official guide](https://github.com/unslothai/unsloth#installation) for your CUDA version, then re-run `pip install -r requirements.txt`.

### 4. Set up Kaggle credentials

The dataset is downloaded via [kagglehub](https://github.com/Kaggle/kagglehub). You need a free Kaggle account:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows)

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

Detected VRAM: 8.0 GB  →  Suggested config: --gpu 8gb
```

### Run training

```bash
# Use the suggested config
python train.py --gpu 8gb

# Override number of steps
python train.py --gpu 16gb --steps 200
```

Training saves LoRA adapters to `lora_australian_law_model/` when complete.

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

Then open **http://localhost:7860** in your browser.

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

- Chat history with multi-turn context
- Adjustable generation settings (max tokens)
- Example prompts to get started
- Jurisdiction and citation-aware system prompt

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
- Close other GPU-intensive applications

**`kagglehub` authentication error**
- Verify `kaggle.json` is in the correct location and has read permissions

**Windows `pickle` / multiprocessing errors**
- The trainer is configured with `dataset_num_proc=1` and `save_strategy="no"` to prevent this. If errors persist, ensure you are running inside a `if __name__ == "__main__":` guard (already handled in `train.py`).

**Gradio port already in use**
- `python serve.py --port 7861`

---

## Disclaimer

This project is for **research and educational purposes only**. The fine-tuned model does not constitute legal advice. Always consult a qualified Australian lawyer for legal matters.

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — fast fine-tuning with minimal VRAM
- [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) — Umar Butler
- [Meta LLaMA](https://llama.meta.com/) — base model
