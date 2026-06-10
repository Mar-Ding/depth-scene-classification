"""Main entry point for CLIP cross-modal sensor adaptation.

Usage:
    python main.py                               # Default: synthetic data
    python main.py --data mat                    # Use NYU .mat file
    python main.py --data synthetic              # Use synthetic data
    python main.py --data mat --num-train 600    # 600 train samples
"""

import argparse
import torch
import numpy as np
import random
import os
from pathlib import Path
import sys

# Set HF mirror for SSL-constrained environments (Windows firewall, proxies)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.models.clip_wrapper import CLIPWrapper
from src.models.depth_processor import DepthProcessor
from src.models.sensor_adapter import MLPAdapter, CrossAttentionAdapter
from src.training.loss import AlignedContrastiveLoss
from src.training.trainer import Trainer
from src.evaluation.zero_shot import ZeroShotEvaluator
from src.visualization.visualize import (
    plot_training_curves,
    plot_accuracy_comparison,
    plot_per_class_accuracy,
    save_results_json,
)
from src.data.synthetic_dataset import create_synthetic_dataloaders


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
    parser.add_argument("--data", type=str, default="synthetic",
                        choices=["synthetic", "mat", "nyu"],
                        help="Data source: synthetic or NYU .mat file")
    parser.add_argument("--mat-path", type=str,
                        default="nyu_depth_v2_labeled.mat",
                        help="Path to NYU .mat file")
    parser.add_argument("--num-train", type=int, default=600,
                        help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=100,
                        help="Number of validation samples")
    parser.add_argument("--num-test", type=int, default=200,
                        help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight decay")
    parser.add_argument("--adapter", type=str, default="mlp",
                        choices=["mlp", "cross_attn"])
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override adapter dropout")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override adapter num_layers")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override contrastive temperature")
    parser.add_argument("--prompt-strategy", type=str, default="simple",
                        choices=["simple", "ensemble", "contrast"],
                        help="Prompt strategy for zero-shot text embeddings")
    return parser


def get_dataloaders(cfg, args):
    """Create data loaders based on user selection."""
    if args.data in ("mat", "nyu"):
        print(f"\n[3/6] Loading NYU Depth V2 from .mat file...")
        from src.data.nyu_mat_dataset import create_dataloaders_from_mat
        return create_dataloaders_from_mat(
            mat_path=args.mat_path,
            num_train=cfg.num_train_samples,
            num_val=cfg.num_val_samples,
            num_test=cfg.num_test_samples,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
    else:
        print(f"\n[3/6] Generating synthetic depth data...")
        return create_synthetic_dataloaders(
            num_train=cfg.num_train_samples,
            num_val=cfg.num_val_samples,
            num_test=cfg.num_test_samples,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            seed=cfg.seed,
        )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Auto-detect: use .mat if file exists
    use_mat = args.data in ("mat", "nyu")
    if use_mat:
        mat_path = Path(args.mat_path)
        if not mat_path.exists():
            print(f"ERROR: .mat file not found at {mat_path}")
            print(f"  Download from: https://horatio.cs.nyu.edu/mit/silberman/")
            print(f"  nyu_depth_v2/nyu_depth_v2_labeled.mat")
            print(f"  Or use --data synthetic to run with synthetic data")
            sys.exit(1)

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

    data_src = "NYU Depth V2 (.mat)" if use_mat else "synthetic"
    print(f"Config: device={cfg.device}, adapter={cfg.adapter_type}, "
          f"samples={cfg.num_train_samples}+{cfg.num_val_samples}+{cfg.num_test_samples}")
    print(f"  Data: {data_src}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply command line overrides
    if args.dropout is not None:
        cfg.adapter_dropout = args.dropout
        print(f"  Override: dropout={cfg.adapter_dropout}")
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
        print(f"  Override: weight_decay={cfg.weight_decay}")
    if args.num_layers is not None:
        cfg.adapter_num_layers = args.num_layers
        print(f"  Override: num_layers={cfg.adapter_num_layers}")
    if args.temperature is not None:
        cfg.temperature = args.temperature
        print(f"  Override: temperature={cfg.temperature}")

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
            input_dim=clip.vision_hidden_dim,  # 768 for ViT-B/32 (patch token dim)
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
        train_loader, val_loader, test_loader = get_dataloaders(cfg, args)
        print(f"  Train: {len(train_loader.dataset)} | "
              f"Val: {len(val_loader.dataset)} | "
              f"Test: {len(test_loader.dataset)}")

        print(f"\n[4/6] Training adapter...")
        trainer = Trainer(adapter, clip, depth_processor, loss_fn, cfg)
        history = trainer.train(train_loader, val_loader)
        plot_training_curves(history)

    if args.mode in ("evaluate", "all"):
        if args.mode != "all":
            _, _, test_loader = get_dataloaders(cfg, args)

        best_path = output_dir / "best_adapter.pt"
        if best_path.exists():
            adapter.load_state_dict(torch.load(best_path, map_location=cfg.device))
            print(f"  Loaded best adapter from {best_path}")
        else:
            print(f"  WARNING: No saved adapter found at {best_path}")

        print(f"\\n[5/6] Zero-shot evaluation...")

        # Get class names — use full set from actual data when available
        class_names = cfg.nyu_classes
        if use_mat:
            try:
                # Use whatever loader is available to access the dataset
                loader = train_loader if args.mode in ("train", "all") else test_loader
                full_dataset = loader.dataset.dataset
                if hasattr(full_dataset, "full_class_names"):
                    class_names = full_dataset.full_class_names
                    print(f"  Using {len(class_names)} scene types from dataset")
            except Exception:
                pass

        evaluator = ZeroShotEvaluator(
            adapter, clip, class_names, device=cfg.device,
            prompt_strategy=args.prompt_strategy,
        )
        results = evaluator.evaluate(test_loader)

        print(f"\n  Results:")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.2%}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.2%}")

        print(f"\n  Computing RGB baseline (CLIP upper bound)...")
        rgb_results = evaluator.evaluate_rgb_baseline(test_loader)
        print(f"  RGB Top-1: {rgb_results['top1_accuracy']:.2%}")

        plot_accuracy_comparison(results, rgb_results)
        plot_per_class_accuracy(results["per_class_accuracy"])
        save_results_json(results)

    if args.mode in ("visualize",):
        print(f"\n[6/6] Generating visualizations from saved data...")
        if (output_dir / "results.json").exists():
            print("  Results already saved, see output/ directory.")
        else:
            print("  No saved results found. Run 'evaluate' mode first.")

    print("\nDone! Check output/ directory for results and visualizations.")


if __name__ == "__main__":
    main()
