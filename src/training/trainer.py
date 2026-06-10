"""Training loop for sensor adaptation."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import json
from pathlib import Path
from src.models.sensor_adapter import CrossAttentionAdapter


class Trainer:
    """Trainer for contrastive sensor adapter learning."""

    def __init__(
        self,
        adapter: nn.Module,
        clip_wrapper: nn.Module,
        depth_processor: nn.Module,
        loss_fn: nn.Module,
        config,
    ):
        self.adapter = adapter
        self.clip = clip_wrapper
        self.depth_processor = depth_processor
        self.loss_fn = loss_fn
        self.config = config
        self.is_cross_attn = isinstance(adapter, CrossAttentionAdapter)

        self.optimizer = AdamW(
            adapter.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        self.device = config.device
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _get_depth_features(self, pixel_values):
        """Get depth features: patch tokens for cross_attn, pooled for mlp."""
        with torch.no_grad():
            if self.is_cross_attn:
                return self.clip.encode_rgb_patch_tokens(pixel_values)  # (B, N, D)
            else:
                return self.clip.encode_rgb(pixel_values)  # (B, D)

    def _apply_adapter(self, features):
        """Apply adapter, handling different return types."""
        if self.is_cross_attn:
            global_emb, local_tokens, attn_weights = self.adapter(features)
            return global_emb
        else:
            return self.adapter(features)

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.adapter.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            rgb_pil = batch["rgb_pil"]
            depth_pil = batch["depth_pil"]

            # Process RGB through CLIP
            rgb_inputs = self.clip.processor(
                images=rgb_pil,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                rgb_features = self.clip.encode_rgb(rgb_inputs["pixel_values"])

            # Process depth through CLIP → adapter
            # First convert depth PIL to CLIP-compatible 3-channel
            depth_tensors = []
            for d in depth_pil:
                d_rgb = d.convert("RGB")  # 1-channel → 3-channel
                depth_tensors.append(d_rgb)

            depth_inputs = self.clip.processor(
                images=depth_tensors,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                depth_clip_features = self._get_depth_features(depth_inputs["pixel_values"])

            # Apply adapter
            adapted = self._apply_adapter(depth_clip_features)

            # Compute loss
            loss = self.loss_fn(adapted, rgb_features)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader) -> tuple:
        """Evaluate on validation set.

        Returns:
            (avg_loss, accuracy)
        """
        self.adapter.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            rgb_pil = batch["rgb_pil"]
            depth_pil = batch["depth_pil"]
            labels = batch["label"].to(self.device)

            # RGB features (CLIP)
            rgb_inputs = self.clip.processor(
                images=rgb_pil,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            rgb_features = self.clip.encode_rgb(rgb_inputs["pixel_values"])

            # Depth → adapter
            depth_tensors = [d.convert("RGB") for d in depth_pil]
            depth_inputs = self.clip.processor(
                images=depth_tensors,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            depth_clip_features = self._get_depth_features(depth_inputs["pixel_values"])
            adapted = self._apply_adapter(depth_clip_features)

            # Loss
            loss = self.loss_fn(adapted, rgb_features)
            total_loss += loss.item()

            # Zero-shot classification via text
            # Get text embeddings for all scene classes
            text_embeds = self.clip.get_class_text_embeds(
                self.config.nyu_classes,
                prefix="a photo of a {}"
            )

            # Adapted depth → text similarity
            sim = adapted @ text_embeds.T  # (B, C)
            preds = sim.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train(self, train_loader, val_loader):
        """Full training loop."""
        best_val_acc = 0.0

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2%}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.adapter.state_dict(),
                           self.output_dir / "best_adapter.pt")

        # Save final model and history
        torch.save(self.adapter.state_dict(),
                   self.output_dir / "final_adapter.pt")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best Val Acc: {best_val_acc:.2%}")
        return self.history
