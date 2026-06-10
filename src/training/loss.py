"""Contrastive loss functions for cross-modal alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    The standard contrastive loss used in CLIP training.
    For a batch of N pairs, creates N×N similarity matrix where
    diagonal elements are positive pairs and off-diagonal are negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss between two sets of embeddings.

        Args:
            z1: (B, D) embeddings from view 1 (e.g., depth→adapter)
            z2: (B, D) embeddings from view 2 (e.g., CLIP text encodings)

        Returns:
            scalar loss
        """
        B = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity matrix: (B, B)
        sim = torch.mm(z1, z2.T) / self.temperature

        # Symmetric loss: average of two directions
        labels = torch.arange(B, device=z1.device)

        loss_1 = F.cross_entropy(sim, labels)  # z1→z2 direction
        loss_2 = F.cross_entropy(sim.T, labels)  # z2→z1 direction

        return (loss_1 + loss_2) / 2.0


class AlignedContrastiveLoss(nn.Module):
    """Contrastive loss between depth (via adapter) and RGB (via CLIP) embeddings.

    Uses RGB as the alignment target — the adapter learns to produce
    depth embeddings that align with the RGB embedding space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.contrastive = NTXentLoss(temperature=temperature)

    def forward(
        self,
        depth_embeds: torch.Tensor,
        rgb_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Align depth embeddings with RGB embeddings.

        Args:
            depth_embeds: (B, D) adapted depth features
            rgb_embeds: (B, D) CLIP RGB features (frozen, no grad)

        Returns:
            scalar loss
        """
        return self.contrastive(depth_embeds, rgb_embeds)
