"""Main entry point for depth scene classification with DINOv2 backbone.

Integrates best-practice setup:
- Data augmentation (RandomHFlip + RandomAffine + ColorJitter)
- Label smoothing cross-entropy loss
- 27 raw → 13 NYU class mapping
- Cosine annealing LR scheduler
- 100 epochs

Usage:
    python main.py                                    # Default: NYU depth
    python main.py --use-rgb                          # RGB baseline
    python main.py --backbone dinov2-base             # Specify backbone
    python main.py --load-depth-weights /path/to.pth  # Depth Anything weights
    python main.py --no-augment                       # Disable augmentation
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from pathlib import Path
import sys
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms as T

# Set HF mirror for environments with restricted network
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.models.depth_anything_wrapper import DepthAnythingWrapper
from src.data.nyu_mat_dataset import NYU13_CLASSES, NYU13_NAMES, NYUDepthMatDataset
from src.visualization.visualize import (
    plot_training_curves,
    plot_accuracy_comparison,
    plot_per_class_accuracy,
    save_results_json,
)


# ── Label smoothing loss ──────────────────────────────────────────────
class LabelSmoothCE(nn.Module):
    def __init__(self, epsilon=0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        smoothed = torch.full_like(logits, self.epsilon / (n_classes - 1))
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smoothed * log_probs).sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ── Build 27→13 label mapping ────────────────────────────────────────
def build_nyu13_mapping(dataset):
    """Build mapping from 27 raw scene type IDs to 13 NYU class IDs."""
    mapping = {}
    for raw_id, aliased_name in dataset.type_names.items():
        if aliased_name in NYU13_CLASSES:
            mapping[raw_id] = NYU13_CLASSES[aliased_name]
        else:
            print(f"  WARNING: '{aliased_name}' (raw_id={raw_id}) "
                  f"not in NYU13, mapping to 0")
            mapping[raw_id] = 0
    return mapping


# ── Data loaders with optional augmentation ──────────────────────────
def create_dataloaders(
    mat_path="nyu_depth_v2_labeled.mat",
    num_train=900, num_val=200, num_test=249,
    batch_size=16, image_size=518, num_workers=0, seed=42,
    augment=True,
):
    """Create train/val/test dataloaders with optional augmentation."""
    # Base transform (resize only)
    base_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        T.ToTensor(),
    ])

    # Augmentation for training (matches paper: RandomHFlip + Affine + ColorJitter)
    if augment:
        train_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.08, 0.08),
                           scale=(0.92, 1.08), fill=0),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
        ])
    else:
        train_transform = base_transform

    # Load full dataset
    full = NYUDepthMatDataset(
        mat_path=mat_path, split="all",
        image_size=image_size, seed=seed,
    )

    # Shuffle and split
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

    # Build label map: 27 raw → 13 NYU classes
    label_map = build_nyu13_mapping(full)
    num_classes = 13
    class_names = [NYU13_NAMES[i] for i in range(num_classes)]

    # Augmented subset wrapper
    class AugmentedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform, label_map):
            self.subset = subset
            self.transform = transform
            self.label_map = label_map

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            item = self.subset[idx]
            depth_pil = item["depth_pil"].convert("RGB")
            depth_tensor = self.transform(depth_pil)
            rgb_pil = item["rgb_pil"].convert("RGB")
            rgb_tensor = base_transform(rgb_pil)
            label = self.label_map[int(item["label"])]
            return {
                "depth_tensor": depth_tensor,
                "rgb_tensor": rgb_tensor,
                "label": label,
                "scene": item["scene"],
            }

    def collate_fn(batch):
        return {
            "depth_tensor": torch.stack([b["depth_tensor"] for b in batch]),
            "rgb_tensor": torch.stack([b["rgb_tensor"] for b in batch]),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "scene": [b["scene"] for b in batch],
        }

    from torch.utils.data import Subset

    train_set = AugmentedSubset(Subset(full, train_idx), train_transform, label_map)
    val_set = AugmentedSubset(Subset(full, val_idx), base_transform, label_map)
    test_set = AugmentedSubset(Subset(full, test_idx), base_transform, label_map)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  Classes: {num_classes} ({', '.join(class_names)})")

    return train_loader, val_loader, test_loader, class_names


# ── Training loop ────────────────────────────────────────────────────
class AugmentedTrainer:
    def __init__(self, classifier, backbone, loss_fn, config):
        self.classifier = classifier
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.config = config
        self.device = config.device
        self.use_rgb = config.use_rgb_input

        self.optimizer = AdamW(
            classifier.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _extract(self, batch):
        if self.use_rgb:
            pixel_values = batch["rgb_tensor"]
        else:
            pixel_values = batch["depth_tensor"]
        with torch.no_grad():
            features = self.backbone.encode_depth(pixel_values.to(self.device))
        return features

    def train_epoch(self, loader):
        self.classifier.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Train", leave=False):
            labels = batch["label"].to(self.device)
            features = self._extract(batch)
            logits = self.classifier(features)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(loader, desc="Val", leave=False):
            labels = batch["label"].to(self.device)
            features = self._extract(batch)
            logits = self.classifier(features)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), correct / total

    def train(self, train_loader, val_loader):
        best_acc = 0.0
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.scheduler.step()

            print(
                f"Ep {epoch+1:3d}/{self.config.num_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Acc: {val_acc:.2%}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.classifier.state_dict(),
                           self.output_dir / "best_classifier.pt")

        torch.save(self.classifier.state_dict(),
                   self.output_dir / "final_classifier.pt")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Best Val Acc: {best_acc:.2%}")
        return self.history


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Depth Scene Classification with DINOv2 Backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "evaluate", "visualize", "all"],
                        help="Pipeline mode")
    parser.add_argument("--backbone", type=str, default="facebook/dinov2-base",
                        choices=["facebook/dinov2-small", "facebook/dinov2-base",
                                 "facebook/dinov2-large"],
                        help="Backbone model")
    parser.add_argument("--load-depth-weights", type=str, default="",
                        help="Path to Depth Anything .pth file")
    parser.add_argument("--use-rgb", action="store_true",
                        help="Use RGB images instead of depth")
    parser.add_argument("--mat-path", type=str,
                        default="nyu_depth_v2_labeled.mat",
                        help="Path to NYU .mat file")
    parser.add_argument("--num-train", type=int, default=900,
                        help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=200,
                        help="Number of validation samples")
    parser.add_argument("--num-test", type=int, default=249,
                        help="Number of test samples (max 249)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Check for .mat file
    mat_path = Path(args.mat_path)
    if not mat_path.exists():
        print(f"ERROR: .mat file not found at {mat_path}")
        print(f"  Download from: https://horatio.cs.nyu.edu/mit/silberman/"
              f"nyu_depth_v2/nyu_depth_v2_labeled.mat")
        sys.exit(1)

    set_seed(args.seed)

    # Config
    cfg = Config(
        backbone_model_name=args.backbone,
        depth_weights_path=args.load_depth_weights,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smooth,
        augment=not args.no_augment,
        num_train_samples=args.num_train,
        num_val_samples=args.num_val,
        num_test_samples=args.num_test,
        output_dir=args.output_dir,
        seed=args.seed,
        use_rgb_input=args.use_rgb,
    )

    mode_str = "RGB baseline" if args.use_rgb else "Depth classification"
    aug_str = "with augmentation" if cfg.augment else "no augmentation"
    print(f"Mode: {mode_str} ({aug_str})")
    print(f"Config: device={cfg.device}, backbone={cfg.backbone_model_name}")
    print(f"  Data: train={cfg.num_train_samples}, "
          f"val={cfg.num_val_samples}, test={cfg.num_test_samples}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Backbone ──
    print(f"\n[1/5] Loading backbone model...")
    depth_weights = cfg.depth_weights_path or None
    backbone = DepthAnythingWrapper(
        model_name=cfg.backbone_model_name,
        device=cfg.device,
        depth_weights_path=depth_weights,
    )
    print(f"  Backbone loaded: {cfg.backbone_model_name} (dim={backbone.feature_dim})")

    # ── 2. Data (13-class NYU mapping) ──
    print(f"\n[2/5] Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        mat_path=str(mat_path),
        num_train=cfg.num_train_samples,
        num_val=cfg.num_val_samples,
        num_test=cfg.num_test_samples,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        augment=cfg.augment,
    )
    num_classes = len(class_names)

    # ── 3. Linear classifier ──
    print(f"\n[3/5] Creating linear classifier ({num_classes} classes)...")
    classifier = nn.Linear(backbone.feature_dim, num_classes).to(cfg.device)
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier params: {classifier_params:,}")

    # ── 4. Loss ──
    loss_fn = LabelSmoothCE(epsilon=cfg.label_smoothing)

    if args.mode in ("train", "all"):
        print(f"  Train: {len(train_loader.dataset)} | "
              f"Val: {len(val_loader.dataset)} | "
              f"Test: {len(test_loader.dataset)}")

        print(f"\n[4/5] Training linear classifier ({cfg.num_epochs} epochs)...")
        trainer = AugmentedTrainer(classifier, backbone, loss_fn, cfg)
        history = trainer.train(train_loader, val_loader)
        plot_training_curves(history)

    if args.mode in ("evaluate", "all"):
        print(f"\n[5/5] Evaluating on test set...")

        best_path = output_dir / "best_classifier.pt"
        if best_path.exists():
            classifier.load_state_dict(torch.load(best_path, map_location=cfg.device))
            print(f"  Loaded best classifier from {best_path}")
        else:
            print(f"  WARNING: No saved classifier found at {best_path}")

        # Test evaluation
        classifier.eval()
        all_preds, all_labels = [], []
        for batch in tqdm(test_loader, desc="Test"):
            labels = batch["label"].to(cfg.device)
            px = batch["rgb_tensor" if args.use_rgb else "depth_tensor"].to(cfg.device)
            with torch.no_grad():
                features = backbone.encode_depth(px)
                logits = classifier(features)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        top1 = (preds == labels).float().mean().item()

        per_class = {}
        for i, name in enumerate(class_names):
            mask = labels == i
            if mask.sum() > 0:
                per_class[name] = (preds[mask] == labels[mask]).float().mean().item()

        print(f"\n  Depth Results:")
        print(f"  Top-1 Accuracy: {top1:.2%}")
        print(f"  Per-class:")
        for name, acc in sorted(per_class.items()):
            if not np.isnan(acc):
                print(f"    {name}: {acc:.2%}")

        if not args.use_rgb:
            print(f"\n  Computing RGB baseline...")
            classifier.eval()
            all_preds_rgb, all_labels_rgb = [], []
            for batch in tqdm(test_loader, desc="RGB"):
                labels = batch["label"].to(cfg.device)
                px = batch["rgb_tensor"].to(cfg.device)
                with torch.no_grad():
                    features = backbone.encode_depth(px)
                    logits = classifier(features)
                preds = logits.argmax(dim=1)
                all_preds_rgb.append(preds.cpu())
                all_labels_rgb.append(labels.cpu())
            preds_rgb = torch.cat(all_preds_rgb)
            labels_rgb = torch.cat(all_labels_rgb)
            rgb_top1 = (preds_rgb == labels_rgb).float().mean().item()
            print(f"  RGB Top-1: {rgb_top1:.2%}")

            results = {"top1_accuracy": top1, "per_class_accuracy": per_class}
            rgb_results = {"top1_accuracy": rgb_top1}
            plot_accuracy_comparison(results, rgb_results)
        else:
            results = {"top1_accuracy": top1, "per_class_accuracy": per_class}

        plot_per_class_accuracy(per_class)
        save_results_json(results)

    if args.mode in ("visualize",):
        print(f"\nGenerating visualizations from saved data...")
        print("  See output/ directory.")

    print("\nDone! Check output/ directory for results and visualizations.")


if __name__ == "__main__":
    main()
