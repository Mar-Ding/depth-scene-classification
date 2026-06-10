"""Main entry point for CLIP cross-modal sensor adaptation.

Usage:
    python main.py                  # Full pipeline: train + evaluate + visualize
    python main.py --mode train     # Train only
    python main.py --mode evaluate  # Evaluate only (loads saved adapter)
    python main.py --mode visualize # Visualize only (uses saved results)
"""

import argparse
import torch
import numpy as np
import random
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.models.clip_wrapper import CLIPWrapper
from src.models.depth_processor import DepthProcessor
from src.models.sensor_adapter import MLPAdapter, CrossAttentionAdapter
from src.data.nyu_dataset import create_nyu_dataloaders
from src.training.loss import AlignedContrastiveLoss
from src.training.trainer import Trainer
from src.evaluation.zero_shot import ZeroShotEvaluator
from src.visualization.visualize import (
    plot_training_curves,
    plot_accuracy_comparison,
    plot_per_class_accuracy,
    save_results_json,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="CLIP Cross-modal Sensor Adaptation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "evaluate", "visualize", "all"],
                        help="Pipeline mode")
    parser.add_argument("--num-train", type=int, default=200,
                        help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=50,
                        help="Number of validation samples")
    parser.add_argument("--num-test", type=int, default=50,
                        help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adapter", type=str, default="mlp",
                        choices=["mlp", "cross_attn"])
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    # Config
    cfg = Config(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        adapter_type=args.adapter,
        num_train_samples=args.num_train,
        num_val_samples=args.num_val,
        num_test_samples=args.num_test,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Config: device={cfg.device}, adapter={cfg.adapter_type}, "
          f"samples={cfg.num_train_samples}+{cfg.num_val_samples}+{cfg.num_test_samples}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP
    print("\n[1/6] Loading CLIP model...")
    clip = CLIPWrapper(model_name=cfg.clip_model_name, device=cfg.device)
    print(f"  CLIP loaded: {cfg.clip_model_name} (dim={clip.feature_dim})")

    # Depth processor
    depth_processor = DepthProcessor(strategy="repeat", image_size=cfg.image_size)

    # Create adapter
    print(f"\n[2/6] Creating adapter ({cfg.adapter_type})...")
    if cfg.adapter_type == "mlp":
        adapter = MLPAdapter(
            input_dim=clip.feature_dim,
            hidden_dim=cfg.adapter_hidden_dim,
            output_dim=clip.feature_dim,
            num_layers=cfg.adapter_num_layers,
            dropout=cfg.adapter_dropout,
        ).to(cfg.device)
    else:
        adapter = CrossAttentionAdapter(
            input_dim=clip.feature_dim,
            hidden_dim=cfg.adapter_hidden_dim,
            output_dim=clip.feature_dim,
            num_queries=cfg.num_query_tokens,
            num_heads=cfg.num_cross_attn_heads,
            dropout=cfg.adapter_dropout,
        ).to(cfg.device)
    print(f"  Adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    # Loss
    loss_fn = AlignedContrastiveLoss(temperature=cfg.temperature)

    if args.mode in ("train", "all"):
        # Data
        print(f"\n[3/6] Loading NYU Depth V2 data...")
        train_loader, val_loader, test_loader = create_nyu_dataloaders(
            num_train=cfg.num_train_samples,
            num_val=cfg.num_val_samples,
            num_test=cfg.num_test_samples,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
        print(f"  Train: {len(train_loader.dataset)} | "
              f"Val: {len(val_loader.dataset)} | "
              f"Test: {len(test_loader.dataset)}")

        # Train
        print(f"\n[4/6] Training adapter...")
        trainer = Trainer(adapter, clip, depth_processor, loss_fn, cfg)
        history = trainer.train(train_loader, val_loader)

        # Plot curves
        plot_training_curves(history)

    if args.mode in ("evaluate", "all"):
        # Load or reuse data
        if args.mode == "all":
            # Reuse test_loader from training
            pass
        else:
            print(f"\n[3/6] Loading data for evaluation...")
            _, _, test_loader = create_nyu_dataloaders(
                num_train=cfg.num_train_samples,
                num_val=cfg.num_val_samples,
                num_test=cfg.num_test_samples,
                batch_size=cfg.batch_size,
                image_size=cfg.image_size,
                seed=cfg.seed,
            )

        # Load best adapter
        best_path = output_dir / "best_adapter.pt"
        if best_path.exists():
            adapter.load_state_dict(torch.load(best_path, map_location=cfg.device))
            print(f"  Loaded best adapter from {best_path}")
        else:
            print(f"  WARNING: No saved adapter found at {best_path}")

        # Evaluate
        print(f"\n[5/6] Zero-shot evaluation...")
        evaluator = ZeroShotEvaluator(
            adapter, clip, cfg.nyu_classes, device=cfg.device,
        )
        results = evaluator.evaluate(test_loader)

        print(f"\n  Results:")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.2%}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.2%}")

        # RGB baseline
        print(f"\n  Computing RGB baseline (CLIP upper bound)...")
        rgb_results = evaluator.evaluate_rgb_baseline(test_loader)
        print(f"  RGB Top-1: {rgb_results['top1_accuracy']:.2%}")

        # Save
        plot_accuracy_comparison(results, rgb_results)
        plot_per_class_accuracy(results["per_class_accuracy"])
        save_results_json(results)

    if args.mode in ("visualize",):
        print(f"\n[6/6] Generating visualizations from saved results...")
        # Just generate visualizations from saved data
        if (output_dir / "results.json").exists():
            print("  Results already saved, see output/ directory.")
        else:
            print("  No saved results found. Run 'evaluate' mode first.")

    print("\nDone! Check output/ directory for results and visualizations.")


if __name__ == "__main__":
    main()
