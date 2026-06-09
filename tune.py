import os
import torch
import gc
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Environment variables to optimize local Windows training execution
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ==========================================
# 0. VRAM Pre-Clean
# ==========================================
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 1. Configuration & Setup
# ==========================================
max_seq_length = 512   
dtype = None           
load_in_4bit = True    

model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

print(f"Loading {model_name} in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

tokenizer.padding_side = "right"

# ==========================================
# 2. Inject LoRA Adapters
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# ==========================================
# 3. Load & Map the Open Australian Legal Corpus
# ==========================================
print("Resolving local Open Australian Legal Corpus path via kagglehub...")
import kagglehub

dataset_dir = kagglehub.dataset_download("umarbutler/open-australian-legal-corpus")
kaggle_dataset_path = os.path.join(dataset_dir, "corpus.jsonl")

print(f"Targeting dataset file directly at: {kaggle_dataset_path}")

dataset = load_dataset("json", data_files=kaggle_dataset_path, split="train")

alpaca_prompt = """Below is an instruction that describes a legal task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the following Australian legal document. Synthesize its legal principles based on the provided jurisdiction and context framework.

### Input:
Citation: {}
Jurisdiction: {}
Document Type: {}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    citations     = examples.get("citation", ["Unknown Citation"] * len(examples["text"]))
    jurisdictions = examples.get("jurisdiction", ["Unknown Jurisdiction"] * len(examples["text"]))
    types         = examples.get("type", ["Unknown Type"] * len(examples["text"]))
    texts         = examples["text"]
    
    formatted_texts = []
    for citation, jurisdiction, doc_type, text in zip(citations, jurisdictions, types, texts):
        raw_prompt = alpaca_prompt.format(citation, jurisdiction, doc_type, text if text else "") + EOS_TOKEN
        
        # Safe boundary to dodge input dimension overflows
        tokens = tokenizer.encode(raw_prompt, truncation=True, max_length=500)
        truncated_prompt = tokenizer.decode(tokens, skip_special_tokens=False)
        formatted_texts.append(truncated_prompt)
        
    return { "text" : formatted_texts, }

print("Formatting and mapping dataset structure with hard token truncation...")
dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. Configure the Trainer (Windows-Pickle Patch Added)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,              
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 1,   
        gradient_accumulation_steps = 8,   
        warmup_steps = 5,
        max_steps = 60,                    
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",              
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "unsloth_australian_legal_lora",
        # --- THE WINDOWS FIX ---
        save_strategy = "no",             # Blocks internal automatic pickling checkpoint saves
        disable_tqdm = False,
    ),
)

# ==========================================
# 5. Start Training & Direct Safe Save
# ==========================================
gc.collect()
torch.cuda.empty_cache()

print("Launching local legal fine-tuning loop...")
trainer_stats = trainer.train()

# Explicit manual saving blocks environment namespace lookup errors entirely
print("Training run finished successfully. Safely exporting fine-tuned LoRA adapters manually...")
model.save_pretrained("lora_australian_law_model")
tokenizer.save_pretrained("lora_australian_law_model")

print("\n========================================================")
print("SUCCESS! Fine-tuned adapters are saved inside: 'lora_australian_law_model'")
print("========================================================")