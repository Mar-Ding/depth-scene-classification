"""Configuration for CLIP cross-modal sensor adaptation."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Config:
    # CLIP model
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # Adapter
    adapter_type: str = "mlp"  # "mlp" or "cross_attn"
    adapter_hidden_dim: int = 512
    adapter_num_layers: int = 2
    adapter_dropout: float = 0.1

    # Cross-attention adapter specific
    num_query_tokens: int = 16
    num_cross_attn_heads: int = 8

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 30
    temperature: float = 0.07

    # Data - small sample mode
    num_train_samples: int = 200
    num_val_samples: int = 50
    num_test_samples: int = 50
    image_size: int = 224
    num_workers: int = 0  # 0 for Windows compatibility

    # Output
    output_dir: str = "./output"
    seed: int = 42

    # Auto-detected device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # NYU Depth V2 scene classes (standard 13-class set)
    nyu_classes: list = field(default_factory=lambda: [
        "bedroom", "living room", "bathroom", "dining room",
        "kitchen", "home office", "office", "classroom",
        "library", "bookstore", "laundry", "furniture store",
        "study"
    ])

    def __post_init__(self):
        """Validate config."""
        assert self.adapter_type in ("mlp", "cross_attn"), \
            f"Unknown adapter type: {self.adapter_type}"
