"""Zero-shot classification evaluation for sensor data."""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Optional
import numpy as np


class ZeroShotEvaluator:
    """Evaluate zero-shot classification performance on depth sensor data.

    Pipeline:
    1. Depth image → CLIP vision encoder → features
    2. Features → Sensor Adapter → adapted embeddings
    3. Adapted embeddings ↔ text class embeddings (cosine similarity)
    4. Top-1 / Top-5 accuracy
    """

    def __init__(
        self,
        adapter: nn.Module,
        clip_wrapper: nn.Module,
        class_names: List[str],
        device: str = "cpu",
    ):
        self.adapter = adapter
        self.clip = clip_wrapper
        self.class_names = class_names
        self.device = device

        # Precompute text embeddings for all classes
        self.text_embeds = clip_wrapper.get_class_text_embeds(
            class_names, prefix="a photo of a {}"
        )  # (C, D)
        print(f"Class text embeddings computed: {self.text_embeds.shape}")

    @torch.no_grad()
    def evaluate(self, test_loader) -> dict:
        """Run zero-shot evaluation on test set.

        Returns:
            dict with accuracy metrics and per-class breakdown
        """
        self.adapter.eval()

        all_preds = []
        all_labels = []
        all_scores = []

        for batch in tqdm(test_loader, desc="Zero-shot eval"):
            depth_pil = batch["depth_pil"]
            labels = batch["label"].to(self.device)

            # Depth → 3-channel → CLIP
            depth_tensors = [d.convert("RGB") for d in depth_pil]
            depth_inputs = self.clip.processor(
                images=depth_tensors,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            depth_features = self.clip.encode_rgb(depth_inputs["pixel_values"])
            adapted = self.adapter(depth_features)  # (B, D)

            # Similarity with text embeddings
            sim = adapted @ self.text_embeds.T  # (B, C)
            scores = torch.softmax(sim, dim=1)

            preds = sim.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        scores = torch.cat(all_scores)

        # Overall accuracy
        top1 = (preds == labels).float().mean().item()
        top5 = self._top5_accuracy(scores, labels)

        # Per-class accuracy
        per_class = {}
        for i, name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                per_class[name] = (preds[mask] == labels[mask]).float().mean().item()
            else:
                per_class[name] = float("nan")

        return {
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "per_class_accuracy": per_class,
            "predictions": preds.numpy(),
            "ground_truth": labels.numpy(),
        }

    def _top5_accuracy(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute top-5 accuracy."""
        top5_preds = scores.topk(5, dim=1).indices
        correct = top5_preds.eq(labels.view(-1, 1)).any(dim=1)
        return correct.float().mean().item()

    @torch.no_grad()
    def evaluate_rgb_baseline(self, test_loader) -> dict:
        """CLIP RGB zero-shot as upper bound reference."""
        all_preds = []
        all_labels = []

        for batch in tqdm(test_loader, desc="RGB baseline"):
            rgb_pil = batch["rgb_pil"]
            labels = batch["label"].to(self.device)

            rgb_inputs = self.clip.processor(
                images=rgb_pil,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            features = self.clip.encode_rgb(rgb_inputs["pixel_values"])

            sim = features @ self.text_embeds.T
            preds = sim.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        top1 = (preds == labels).float().mean().item()

        return {"top1_accuracy": top1}
