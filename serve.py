"""
Australian Law LLM - Local Web UI
===================================
Serves your fine-tuned model as a Gradio chat interface on localhost.

Usage:
    python serve.py                                    # uses default lora_australian_law_model/
    python serve.py --model lora_australian_law_model  # explicit path
    python serve.py --model lora_australian_law_model --port 7860
    python serve.py --merged                           # load a fully merged model instead of LoRA
"""

import argparse
import gc
import os

# Pre-warm triton and torch._inductor before unsloth is imported.
# On Windows, unsloth crashes silently if these are not already cached
# in sys.modules when it loads. This is a known Windows/triton ordering issue.
try:
    import triton
    from torch._inductor.runtime.hints import DeviceProperties
except Exception:
    pass
import torch
import gradio as gr
from unsloth import FastLanguageModel

SYSTEM_PROMPT = """\
You are an expert Australian legal assistant. You have been fine-tuned on the Open Australian Legal Corpus, which includes legislation, case law, and legal instruments across all Australian jurisdictions.

When responding:
- Cite relevant Australian legislation, regulations, or case law where applicable.
- Specify the jurisdiction (Commonwealth, NSW, VIC, QLD, WA, SA, TAS, ACT, NT) when relevant.
- Clarify when a matter requires professional legal advice.
- Be precise and use proper legal terminology.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Serve fine-tuned Australian Law LLM via Gradio")
    parser.add_argument(
        "--model",
        default="lora_australian_law_model",
        help="Path to the LoRA adapter directory (default: lora_australian_law_model)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name/path (auto-detected from adapter config if omitted)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for generation (default: 2048)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port to serve on (default: 7860)",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Load a fully merged model (no LoRA adapter merging needed)",
    )
    return parser.parse_args()


def load_model(args):
    gc.collect()
    torch.cuda.empty_cache()

    model_path = args.model
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: '{model_path}'\n"
            "Run 'python train.py' first to generate the fine-tuned adapters."
        )

    print(f"Loading model from '{model_path}'...")

    if args.merged:
        # Fully merged model — load directly
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        # LoRA adapter — need the base model to attach to
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(
                f"No adapter_config.json found in '{model_path}'. "
                "Make sure training completed successfully."
            )

        import json
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model = args.base_model or adapter_cfg.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("Could not determine base model. Use --base-model to specify it.")

        print(f"Base model : {base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    print("Model ready.\n")
    return model, tokenizer


def build_chat_prompt(tokenizer, history: list, user_message: str) -> str:
    """Build a prompt in the Alpaca instruction format, incorporating chat history.
    history is a list of (user_msg, bot_msg) tuples (Gradio 6 default format).
    """
    history_text = ""
    for user_turn, bot_turn in history[-3:]:   # keep last 3 exchanges as context
        if user_turn:
            history_text += f"\nUser: {user_turn}"
        if bot_turn:
            history_text += f"\nAssistant: {bot_turn}"

    alpaca_prompt = """\
Below is an instruction that describes a legal task. Write a response that appropriately completes the request.

### Instruction:
{system}

### Previous conversation:{history}

### Current question:
{question}

### Response:
"""
    return alpaca_prompt.format(
        system=SYSTEM_PROMPT.strip(),
        history=history_text if history_text else " (none)",
        question=user_message,
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def create_ui(model, tokenizer):
    def clear_history():
        return [], []

    with gr.Blocks(title="Australian Law LLM") as demo:
        gr.Markdown(
            """
# 🇦🇺 Australian Law LLM
**Fine-tuned on the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus)**

Ask questions about Australian legislation, case law, legal principles, or specific jurisdictions.

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It does not constitute legal advice. Consult a qualified Australian lawyer for legal matters.
            """
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=520,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask about Australian law, e.g. 'What are the elements of negligence under Australian common law?'",
                label="Your question",
                scale=8,
                lines=2,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear conversation", variant="secondary")

        with gr.Accordion("Generation settings", open=False):
            max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max new tokens")

        # Wire events — history is list of (user, bot) tuples in Gradio 6
        def chat_with_tokens(user_message, history, max_tok):
            if not user_message.strip():
                return "", history
            prompt = build_chat_prompt(tokenizer, history, user_message)
            response = generate_response(model, tokenizer, prompt, max_new_tokens=max_tok)
            history = history + [(user_message, response)]
            return "", history

        msg.submit(chat_with_tokens, [msg, chatbot, max_tokens], [msg, chatbot])
        submit_btn.click(chat_with_tokens, [msg, chatbot, max_tokens], [msg, chatbot])
        clear_btn.click(clear_history, outputs=[msg, chatbot])

        gr.Examples(
            examples=[
                ["What are the elements of negligence under Australian common law?"],
                ["Explain the doctrine of separation of powers in the Australian Constitution."],
                ["What is the difference between a deed and a contract in Australian law?"],
                ["Summarise the key provisions of the Fair Work Act 2009 (Cth)."],
                ["What constitutes misleading or deceptive conduct under the Australian Consumer Law?"],
            ],
            inputs=msg,
        )

    return demo


def main():
    args = parse_args()
    model, tokenizer = load_model(args)
    demo = create_ui(model, tokenizer)

    print(f"\n{'='*56}")
    print(f"  Australian Law LLM Web UI")
    print(f"  Open in browser:  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*56}\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=".gradio-container { max-width: 900px !important; margin: auto; } footer { display: none !important; }",
    )


if __name__ == "__main__":
    main()
