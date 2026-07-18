"""
Trajectory Trainer — geometric regularisation for LoRA fine-tuning
====================================================================
Subclasses TRL's SFTTrainer to add a trajectory penalty on top of the
standard next-token-prediction loss.

What is trajectory loss?
------------------------
A transformer builds its answer by passing a token's representation
through N layers.  Each layer nudges the hidden state a little — like
steps along a path.  If a single layer makes a huge jump (large L2
displacement) the model is relying on one layer to do disproportionate
work, which correlates with hallucination and brittleness.

The penalty is the mean L2 norm of the displacement between every pair
of consecutive hidden states, averaged across all layer transitions:

    trajectory_loss = mean_over_layers( ||h_{i+1} - h_i||_2 )

The total loss is then:

    total_loss = ntp_loss + trajectory_alpha * trajectory_loss

A small alpha (default 0.01) keeps the regulariser as a soft nudge
rather than dominating the training signal.

VRAM note
---------
output_hidden_states=True retains all intermediate activations in
memory during the forward pass.  gradient_checkpointing should be
enabled (already set via Unsloth's use_gradient_checkpointing="unsloth"
and the explicit gradient_checkpointing=True in TrainingArguments) to
trade recomputation for memory.  If you hit OOM, lower trajectory_alpha
or reduce max_seq_length.
"""

import torch
from trl import SFTTrainer


class TrajectoryTrainer(SFTTrainer):
    """SFTTrainer augmented with a trajectory regularisation loss."""

    def __init__(self, *args, trajectory_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_alpha = trajectory_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Request intermediate hidden states from the forward pass
        inputs = dict(inputs)
        inputs["output_hidden_states"] = True

        outputs = model(**inputs)
        standard_loss = outputs.loss

        # ── Trajectory penalty ──────────────────────────────────────
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
        if hidden_states is not None and len(hidden_states) > 1:
            num_transitions = len(hidden_states) - 1
            displacement_sum = sum(
                torch.norm(hidden_states[i + 1] - hidden_states[i], p=2, dim=-1).mean()
                for i in range(num_transitions)
            )
            trajectory_loss = displacement_sum / num_transitions
        else:
            trajectory_loss = torch.tensor(0.0, device=standard_loss.device)

        total_loss = standard_loss + self.trajectory_alpha * trajectory_loss

        return (total_loss, outputs) if return_outputs else total_loss
