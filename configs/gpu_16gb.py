"""
16 GB VRAM Configuration
Target GPUs: RTX 4080, RTX 3080 Ti, RTX A4000, RX 7900 GRE
Use case: High-quality fine-tuning. Larger rank, longer sequences.
"""

CONFIG = {
    # Model
    "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "max_seq_length": 1024,
    "load_in_4bit": True,
    "dtype": None,

    # LoRA
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 120,
    "learning_rate": 2e-4,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,

    # Dataset
    "dataset_num_proc": 2,
    "packing": False,
    "token_truncation_length": 1000,

    # Output
    "output_dir": "unsloth_australian_legal_lora",
    "lora_save_dir": "lora_australian_law_model",
}
