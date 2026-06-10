"""NYU Depth V2 dataset loader for cross-modal sensor adaptation."""

import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from datasets import load_dataset
from PIL import Image
from typing import Optional, List, Tuple
import random


# Scene label mapping for NYU Depth V2 labeled subset
# The standard 13-class subset used in literature
NYU_SCENE_LABELS = {
    0: "bedroom",
    1: "living room",
    2: "bathroom",
    3: "dining room",
    4: "kitchen",
    5: "home office",
    6: "office",
    7: "classroom",
    8: "library",
    9: "bookstore",
    10: "laundry",
    11: "furniture store",
    12: "study",
}


class NYUDepthDataset(Dataset):
    """NYU Depth V2 dataset returning RGB-Depth pairs with scene labels.

    Uses the labeled subset (1449 RGB-Depth pairs with 13 scene classes).
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 224,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.image_size = image_size
        self.split = split
        self.seed = seed

        # Load NYU Depth V2 from HuggingFace datasets
        ds = load_dataset(
            "nyu_depth_v2",
            split=split,
            trust_remote_code=True,
        )

        # Filter: only keep samples with scene labels
        def has_scene(x):
            return x.get("scene") is not None and len(x.get("scene", "")) > 0

        ds = ds.filter(has_scene)

        # Subset for small sample training
        if num_samples is not None and num_samples < len(ds):
            rng = random.Random(seed)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            ds = ds.select(indices[:num_samples])

        self.dataset = ds
        self.label_map = {v: k for k, v in NYU_SCENE_LABELS.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]

        # RGB image: (H, W, 3) uint8
        rgb = np.array(sample["image"])
        # Depth: (H, W) float32 in meters
        depth = np.array(sample["depth"])
        # Scene label
        scene = sample.get("scene", "")
        label = self.label_map.get(scene, -1)

        # Convert to PIL for consistent processing
        rgb_pil = Image.fromarray(rgb).resize((self.image_size, self.image_size), Image.BILINEAR)
        depth_pil = Image.fromarray(
            ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((self.image_size, self.image_size), Image.BILINEAR)

        return {
            "rgb": rgb_pil,
            "depth": depth_pil,
            "scene": scene,
            "label": label,
            "depth_raw": torch.from_numpy(depth).float(),
        }


def create_nyu_dataloaders(
    num_train: int = 200,
    num_val: int = 50,
    num_test: int = 50,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create train/val/test dataloaders for NYU Depth V2.

    Uses the labeled subset, shuffled and split.
    """
    # Load full labeled dataset
    full_dataset = NYUDepthDataset(
        split="train",
        image_size=image_size,
        num_samples=None,  # load all labeled samples (1449)
        seed=seed,
    )

    # Shuffle indices
    rng = random.Random(seed)
    indices = list(range(len(full_dataset)))
    rng.shuffle(indices)

    # Split
    total_needed = num_train + num_val + num_test
    if total_needed > len(full_dataset):
        print(f"Warning: Requested {total_needed} samples but only {len(full_dataset)} available."
              f" Reducing to available.")
        ratio = len(full_dataset) / total_needed
        num_train = int(num_train * ratio)
        num_val = int(num_val * ratio)
        num_test = int(num_test * ratio)
        # Adjust for rounding
        remaining = len(full_dataset) - num_train - num_val - num_test
        num_test += remaining

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:num_train + num_val + num_test]

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)

    def collate_fn(batch):
        rgb_imgs = [item["rgb"] for item in batch]
        depth_imgs = [item["depth"] for item in batch]
        scenes = [item["scene"] for item in batch]
        labels = [item["label"] for item in batch]
        depth_raw = torch.stack([item["depth_raw"] for item in batch])

        return {
            "rgb_pil": rgb_imgs,
            "depth_pil": depth_imgs,
            "scene": scenes,
            "label": torch.tensor(labels, dtype=torch.long),
            "depth_raw": depth_raw,
        }

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
