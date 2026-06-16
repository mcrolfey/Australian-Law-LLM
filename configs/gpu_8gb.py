"""
8 GB VRAM Configuration
Target GPUs: RTX 3070, RTX 3060 8 GB, RTX 2080, RX 6700 XT
Use case: Balanced fine-tuning. 3B model fits comfortably at 4-bit.
"""

CONFIG = {
    # Model
    "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "max_seq_length": 512,
    "load_in_4bit": True,
    "dtype": None,

    # LoRA
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 50,
    "max_steps": 1000,
    "learning_rate": 5e-5,
    "optim": "adamw_8bit",
    "weight_decay": 0.1,
    "lr_scheduler_type": "cosine",
    "seed": 3407,

    # Logging
    "logging_steps": 10,

    # Dataset
    "dataset_num_proc": 1,
    "packing": False,
    "token_truncation_length": 500,

    # Output
    "output_dir": "unsloth_australian_legal_lora",
    "lora_save_dir": "lora_australian_law_model",
}
