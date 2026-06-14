"""Configuration for depth scene classification with frozen backbone."""

from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # Backbone model
    backbone_model_name: str = "facebook/dinov2-base"
    depth_weights_path: str = ""  # Path to Depth Anything .pth (empty = plain DINOv2)

    # Linear classifier
    classifier_hidden_dim: int = 0  # 0 = linear (no hidden layer), >0 = MLP
    classifier_dropout: float = 0.0

    # Training
    batch_size: int = 16
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100

    # Loss
    label_smoothing: float = 0.1  # 0.0 = standard CE

    # Data augmentation
    augment: bool = True
    aug_hflip_prob: float = 0.5
    aug_affine_degrees: float = 10.0
    aug_affine_translate: float = 0.08
    aug_affine_scale_min: float = 0.92
    aug_affine_scale_max: float = 1.08
    aug_color_jitter_brightness: float = 0.1
    aug_color_jitter_contrast: float = 0.1

    # Label mapping: True = map 27 raw types → 13 NYU semantic classes
    # False = use raw scene type IDs (usually 13+ types)
    use_nyu13_mapping: bool = True

    # Data
    num_train_samples: int = 900
    num_val_samples: int = 200
    num_test_samples: int = 249
    image_size: int = 518  # DINOv2 native size
    num_workers: int = 0  # 0 for Windows compatibility

    # Input source
    use_rgb_input: bool = False  # True = use RGB image, False = use depth map

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
