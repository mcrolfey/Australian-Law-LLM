"""
24 GB VRAM Configuration
Target GPUs: RTX 3090, RTX 4090, RTX A5000, RTX 6000 Ada
Use case: Maximum quality. Larger model, full-length sequences, higher rank.
"""

CONFIG = {
    # Model
    "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "dtype": None,

    # LoRA
    "r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 200,
    "learning_rate": 2e-4,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,

    # Dataset
    "dataset_num_proc": 4,
    "packing": False,
    "token_truncation_length": 2000,

    # Output
    "output_dir": "unsloth_australian_legal_lora",
    "lora_save_dir": "lora_australian_law_model",
}
