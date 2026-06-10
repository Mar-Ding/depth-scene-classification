"""NYU Depth V2 dataset loader from official .mat file.

Supports both v5/v7 .mat (via scipy.io) and v7.3 .mat (via h5py).
"""

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

# Name aliases that appear in NYU v7.3 scene labels
SCENE_ALIASES = {
    "living_room": "living room",
    "dining_room": "dining room",
    "home_office": "home office",
    "laundry_room": "laundry",
    "furniture_store": "furniture store",
    "study_room": "study",
    "computer_lab": "classroom",
    "conference_room": "classroom",
    "dinette": "dining room",
    "indoor_balcony": "living room",
    "office_kitchen": "kitchen",
    "playroom": "living room",
    "reception_room": "living room",
    "student_lounge": "classroom",
    "excercise_room": "living room",
    "home_storage": "bedroom",
    "printer_room": "office",
    "cafe": "dining room",
    "basement": "bedroom",
    "foyer": "living room",
}


def _decode_h5py_str(arr: np.ndarray) -> str:
    """Decode a h5py object reference string (uint16 char codes) to a Python string."""
    chars = arr.flatten()
    return "".join(chr(int(c)) for c in chars).strip()


def _read_scene_types_v73(f, scene_type_refs):
    """Read sceneTypes from v7.3 .mat file (text labels -> numeric IDs)."""
    type_map = {}
    next_id = 0
    labels = []
    for ref in scene_type_refs:
        s = _decode_h5py_str(f[ref][()])
        if s not in type_map:
            type_map[s] = next_id
            next_id += 1
        labels.append(type_map[s])
    return np.array(labels, dtype=np.int32), type_map


class NYUDepthMatDataset(Dataset):
    """NYU Depth V2 from the official labeled .mat file.

    Supports both v5/v7 (scipy) and v7.3 (h5py) format .mat files.
    The .mat file contains 1449 RGB-Depth pairs with scene labels.
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

        # Try h5py first (for v7.3 files), fall back to scipy.io
        try:
            import h5py
            f = h5py.File(mat_path, "r")
            # h5py stores as (N, C, H, W) for v7.3
            images = f["images"][()]  # (1449, 3, 640, 480) uint8
            depths = f["depths"][()]  # (1449, 640, 480) float64 -> float32
            depths = depths.astype(np.float32)
            # Transpose images: (N, C, H, W) -> (N, H, W, C)
            images = images.transpose(0, 2, 3, 1)  # (1449, 640, 480, 3)

            # Read sceneTypes as text labels, map to numeric IDs
            st_refs = f["sceneTypes"][0, :]  # (1449,) of object refs
            scene_types, type_map = _read_scene_types_v73(f, st_refs)

            # Read scene names
            scene_refs = f["scenes"][0, :]
            scene_names = []
            for ref in scene_refs:
                scene_names.append(_decode_h5py_str(f[ref][()]))

            print(f"  Format: v7.3 (h5py), {len(type_map)} scene types found")
            print(f"  Scene types: {sorted(type_map.keys())}")
            f.close()

        except (ImportError, TypeError, AttributeError):
            # Fallback to scipy.io for v5/v7 .mat files
            import scipy.io as sio
            data = sio.loadmat(mat_path)

            # Images: (H, W, 3, N) uint8 -> (N, H, W, 3)
            images = data["images"].transpose(3, 0, 1, 2)  # (1449, 480, 640, 3)

            # Depths: (H, W, N) float64 -> (N, H, W)
            depths = data["depths"].transpose(2, 0, 1).astype(np.float32)  # (1449, 480, 640)

            # Scene types: (1, N) -> (N,)
            scene_types = data["sceneTypes"].flatten().astype(int)  # (1449,)
            # sceneTypes values: 1-13 (0 = unlabeled)
            scene_types = scene_types - 1

            # Scene names
            raw_names = data["names"]
            scene_names = []
            for cell in raw_names.flatten():
                name = str(cell[0]).strip().lower()
                scene_names.append(name)

            type_map = {v: k for k, v in NYU13_CLASSES.items()}
            print(f"  Format: v5/v7 (scipy.io)")

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

        # Store full ordered class names (original scene type names, NOT aliased)
        # Used for zero-shot evaluation with all available scene types
        full_class_names = sorted(type_map.keys(), key=lambda k: type_map[k])
        self.full_class_names = full_class_names

        # Build type_names mapping (for backward compat / aliased display)
        unique_types = sorted(set(self.scene_types.tolist()))
        type_names = {}
        for t in unique_types:
            # Try to match via NYU13
            raw_name = full_class_names[t] if t < len(full_class_names) else None
            matched = False
            for nid_name, nid in NYU13_CLASSES.items():
                if nid == t:
                    type_names[t] = nid_name
                    matched = True
                    break
            if not matched and raw_name:
                aliased = SCENE_ALIASES.get(raw_name, raw_name)
                type_names[t] = aliased
            elif not matched:
                type_names[t] = f"class_{t}"
        self.type_names = type_names
        print(f"  Scene types: {len(unique_types)} unique")
        for t in sorted(type_names.keys()):
            cnt = int((self.scene_types == t).sum())
            print(f"    [{t}] {type_names[t]}: {cnt} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        rgb = self.images[idx]  # (H, W, 3) uint8
        depth = self.depths[idx]  # (H, W) float32
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
    """Create train/val/test dataloaders from the .mat file."""
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
