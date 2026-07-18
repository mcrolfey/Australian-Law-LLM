"""
4 GB VRAM Configuration
Target GPUs: GTX 1650, RTX 3050, RTX 2060 (4 GB), Intel Arc A380
Use case: Minimum viable fine-tuning. Shortest sequences, smallest model.
"""

CONFIG = {
    # Model
    "model_name": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "max_seq_length": 256,
    "load_in_4bit": True,
    "dtype": None,

    # LoRA
    "r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],

    # Trajectory regularisation (geometric penalty on layer-to-layer hidden state displacement)
    "trajectory_alpha": 0.01,

    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
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
    "token_truncation_length": 240,

    # Output
    "output_dir": "unsloth_australian_legal_lora",
    "lora_save_dir": "lora_australian_law_model",
}
