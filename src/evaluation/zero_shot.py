"""Zero-shot classification evaluation for sensor data."""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Optional
import numpy as np
from src.models.sensor_adapter import CrossAttentionAdapter


# Multiple prompt templates for ensemble strategy
ENSEMBLE_TEMPLATES = [
    "a photo of a {}",
    "an indoor scene of a {}",
    "a photo of a {} room",
    "this is a {}",
    "a view of a {}",
]


class ZeroShotEvaluator:
    """Evaluate zero-shot classification performance on depth sensor data.

    Pipeline:
    1. Depth image → CLIP vision encoder → features
    2. Features → Sensor Adapter → adapted embeddings
    3. Adapted embeddings ↔ text class embeddings (cosine similarity)
    4. Top-1 / Top-5 accuracy

    Supports multiple prompt strategies:
    - simple: "a photo of a {class}" (single prompt)
    - ensemble: average over multiple prompt templates
    - contrast: positive - negative prompt logit calibration
    """

    def __init__(
        self,
        adapter: nn.Module,
        clip_wrapper: nn.Module,
        class_names: List[str],
        device: str = "cpu",
        prompt_strategy: str = "simple",
    ):
        self.adapter = adapter
        self.clip = clip_wrapper
        self.class_names = class_names
        self.device = device
        self.is_cross_attn = isinstance(adapter, CrossAttentionAdapter)
        self.prompt_strategy = prompt_strategy

        # Precompute text embeddings
        self.text_embeds = self._build_text_embeds(class_names)
        print(f"Class text embeddings computed: {self.text_embeds.shape}")
        if prompt_strategy != "simple":
            print(f"  Strategy: {prompt_strategy}")

    def _build_text_embeds(self, class_names: List[str]) -> torch.Tensor:
        """Build text embeddings based on prompt strategy."""
        if self.prompt_strategy == "simple":
            return self.clip.get_class_text_embeds(
                class_names, prefix="a photo of a {}"
            )

        elif self.prompt_strategy == "ensemble":
            all_embeds = []
            for tpl in ENSEMBLE_TEMPLATES:
                emb = self.clip.get_class_text_embeds(class_names, prefix=tpl)
                all_embeds.append(emb)
            # Average across templates
            return torch.stack(all_embeds).mean(dim=0)

        elif self.prompt_strategy == "contrast":
            # Positive: "a photo of a {class}"
            pos = self.clip.get_class_text_embeds(
                class_names, prefix="a photo of a {}"
            )
            # Negative: "not a photo of a {class}"
            neg = self.clip.get_class_text_embeds(
                class_names, prefix="not a photo of a {}"
            )
            # Stack: shape (2, num_classes, D) — positive at index 0, negative at 1
            return torch.stack([pos, neg], dim=0)

        else:
            raise ValueError(f"Unknown prompt strategy: {self.prompt_strategy}")

    def _compute_similarity(self, features, text_embeds):
        """Compute similarity based on prompt strategy.

        For 'contrast': sim = sim(pos) - sim(neg)
        For others: standard dot product.
        """
        if self.prompt_strategy == "contrast":
            pos_sim = features @ text_embeds[0].T  # (B, C)
            neg_sim = features @ text_embeds[1].T  # (B, C)
            return pos_sim - neg_sim
        else:
            return features @ text_embeds.T  # (B, C)

    def _get_depth_features(self, pixel_values):
        with torch.no_grad():
            if self.is_cross_attn:
                return self.clip.encode_rgb_patch_tokens(pixel_values)
            else:
                return self.clip.encode_rgb(pixel_values)

    def _apply_adapter(self, features):
        if self.is_cross_attn:
            global_emb, local_tokens, attn_weights = self.adapter(features)
            return global_emb
        else:
            return self.adapter(features)

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

            depth_features = self._get_depth_features(depth_inputs["pixel_values"])
            adapted = self._apply_adapter(depth_features)  # (B, D)

            # Similarity with text embeddings
            sim = self._compute_similarity(adapted, self.text_embeds)
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

            sim = self._compute_similarity(features, self.text_embeds)
            preds = sim.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        top1 = (preds == labels).float().mean().item()

        return {"top1_accuracy": top1}
