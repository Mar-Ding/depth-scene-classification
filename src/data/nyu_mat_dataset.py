"""NYU Depth V2 dataset loader from official .mat file."""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple
import os
import random


# Scene type mapping (13 standard categories from literature)
NYU13_CLASSES = {
    "bedroom": 0, "living room": 1, "bathroom": 2, "dining room": 3,
    "kitchen": 4, "home office": 5, "office": 6, "classroom": 7,
    "library": 8, "bookstore": 9, "laundry": 10, "furniture store": 11,
    "study": 12,
}

# Reverse mapping
NYU13_NAMES = {v: k for k, v in NYU13_CLASSES.items()}


class NYUDepthMatDataset(Dataset):
    """NYU Depth V2 from the official labeled .mat file.

    The .mat file contains 1449 RGB-Depth pairs with scene labels.
    Standard split: train=795, test=654.
    """

    def __init__(
        self,
        mat_path: str = "nyu_depth_v2_labeled.mat",
        split: str = "train",
        image_size: int = 224,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.image_size = image_size
        self.split = split
        self.seed = seed

        if not os.path.exists(mat_path):
            raise FileNotFoundError(
                f"NYU Depth V2 .mat not found at {mat_path}. "
                f"Download from: https://horatio.cs.nyu.edu/mit/silberman/"
                f"nyu_depth_v2/nyu_depth_v2_labeled.mat"
            )

        print(f"Loading .mat file from {mat_path}...")
        import scipy.io as sio
        data = sio.loadmat(mat_path)

        # Images: (H, W, 3, N) uint8 -> (N, H, W, 3)
        images = data["images"].transpose(3, 0, 1, 2)  # (1449, 480, 640, 3)

        # Depths: (H, W, N) float64 -> (N, H, W)
        depths = data["depths"].transpose(2, 0, 1).astype(np.float32)  # (1449, 480, 640)

        # Scene types: (1, N) -> (N,)
        scene_types = data["sceneTypes"].flatten().astype(int)  # (1449,)
        # sceneTypes values: 1-13 (0 = unlabeled)
        # Map to 0-based: scene_types - 1
        scene_types = scene_types - 1

        # Scene names: list of lists (N x 1), extract strings
        raw_names = data["names"]
        scene_names = []
        for cell in raw_names.flatten():
            name = str(cell[0]).strip().lower()
            scene_names.append(name)

        self.images = images
        self.depths = depths
        self.scene_types = scene_types
        self.scene_names = scene_names

        # Filter: keep only labeled samples (scene_type >= 0)
        valid_mask = scene_types >= 0
        self.images = self.images[valid_mask]
        self.depths = self.depths[valid_mask]
        self.scene_types = self.scene_types[valid_mask]
        self.scene_names = [self.scene_names[i] for i in range(len(valid_mask)) if valid_mask[i]]
        print(f"  Loaded {len(self.images)} labeled samples (from {len(valid_mask)} total)")

        # Get unique scene type names
        unique_types = sorted(set(self.scene_types))
        type_names = {}
        for t in unique_types:
            # Find first occurrence to get the name
            idx = list(self.scene_types).index(t)
            type_names[t] = self.scene_names[idx]
            if t in NYU13_CLASSES.values():
                for k, v in NYU13_CLASSES.items():
                    if v == t:
                        type_names[t] = k
        self.type_names = type_names
        print(f"  Scene types: {len(unique_types)} unique")
        for t in sorted(type_names.keys()):
            cnt = list(self.scene_types).count(t)
            print(f"    [{t}] {type_names[t]}: {cnt} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        rgb = self.images[idx]  # (480, 640, 3) uint8
        depth = self.depths[idx]  # (480, 640) float32
        scene_type = self.scene_types[idx]
        scene_name = self.type_names.get(scene_type, f"class_{scene_type}")

        # Resize
        rgb_pil = Image.fromarray(rgb).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        # Normalize depth to 0-255 for PIL
        d_min, d_max = depth.min(), depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_norm).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )

        return {
            "rgb_pil": rgb_pil,
            "depth_pil": depth_pil,
            "scene": scene_name,
            "label": scene_type,
            "depth_raw": torch.from_numpy(depth).float(),
        }


def create_dataloaders_from_mat(
    mat_path: str = "nyu_depth_v2_labeled.mat",
    num_train: int = 600,
    num_val: int = 100,
    num_test: int = 200,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create train/val/test dataloaders from the .mat file.

    Standard split: 795 train / 654 test.
    We further split train into train+val.
    """
    full = NYUDepthMatDataset(
        mat_path=mat_path, split="all",
        image_size=image_size, seed=seed,
    )

    # Shuffle indices
    indices = list(range(len(full)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    total_needed = num_train + num_val + num_test
    if total_needed > len(indices):
        ratio = len(indices) / total_needed
        num_train = int(num_train * ratio)
        num_val = int(num_val * ratio)
        num_test = len(indices) - num_train - num_val

    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train + num_val]
    test_idx = indices[num_train + num_val:num_train + num_val + num_test]

    from torch.utils.data import Subset

    def collate_fn(batch):
        return {
            "rgb_pil": [item["rgb_pil"] for item in batch],
            "depth_pil": [item["depth_pil"] for item in batch],
            "scene": [item["scene"] for item in batch],
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "depth_raw": torch.stack([item["depth_raw"] for item in batch]),
        }

    train_loader = torch.utils.data.DataLoader(
        Subset(full, train_idx), batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        Subset(full, val_idx), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        Subset(full, test_idx), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers
    )

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader
