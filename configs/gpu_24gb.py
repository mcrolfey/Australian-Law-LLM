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

    # Trajectory regularisation (geometric penalty on layer-to-layer hidden state displacement)
    "trajectory_alpha": 0.01,

    # Training — batch halved from 2→1, grad_accum doubled from 4→8 to keep the same
    # effective batch size (8) while reducing peak VRAM from output_hidden_states=True.
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
    "dataset_num_proc": 4,
    "packing": False,
    "token_truncation_length": 2000,

    # Output
    "output_dir": "unsloth_australian_legal_lora",
    "lora_save_dir": "lora_australian_law_model",
}
