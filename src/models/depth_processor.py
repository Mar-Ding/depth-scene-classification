"""Depth map preprocessing utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DepthProcessor(nn.Module):
    """Convert single-channel depth maps to 3-channel representations.

    Supports multiple conversion strategies:
    - repeat: repeat single channel 3 times (simple, preserves raw values)
    - colormap: apply jet colormap for visualization-friendly encoding
    - normal: compute surface normals from depth (3 channels)
    """

    def __init__(self, strategy: str = "repeat", image_size: int = 224):
        super().__init__()
        assert strategy in ("repeat", "colormap", "normal"), f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.image_size = image_size

    def forward(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Process depth maps.

        Args:
            depth_maps: (B, 1, H, W) depth values

        Returns:
            (B, 3, H', W') processed depth as 3-channel image
        """
        if self.strategy == "repeat":
            return self._repeat_channel(depth_maps)
        elif self.strategy == "colormap":
            return self._apply_colormap(depth_maps)
        elif self.strategy == "normal":
            return self._compute_normals(depth_maps)

    def _repeat_channel(self, depth: torch.Tensor) -> torch.Tensor:
        """Repeat single channel 3 times."""
        # Normalize to [0, 1]
        b, c, h, w = depth.shape
        depth_flat = depth.view(b, -1)
        d_min = depth_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        d_max = depth_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        d_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        three_ch = d_norm.repeat(1, 3, 1, 1)
        return F.interpolate(three_ch, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)

    def _apply_colormap(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply jet colormap to depth. Returns 3-channel RGB."""
        # Normalize depth to [0, 1]
        b, c, h, w = depth.shape
        depth_np = depth.detach().cpu().numpy()

        out_maps = []
        for i in range(b):
            d = depth_np[i, 0]
            # Min-max normalize
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            # Apply matplotlib colormap
            import matplotlib.cm as cm
            colored = cm.jet(d_norm)[:, :, :3]  # (H, W, 3)
            out_maps.append(torch.from_numpy(colored).permute(2, 0, 1))

        result = torch.stack(out_maps, dim=0).to(depth.device)
        return F.interpolate(result, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)

    def _compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from depth (3-channel output)."""
        # Normalize depth first
        b, c, h, w = depth.shape
        depth_flat = depth.view(b, -1)
        d_min = depth_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        d_max = depth_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        d_norm = (depth - d_min) / (d_max - d_min + 1e-8)

        # Compute gradients as normals approximation
        grad_y = torch.zeros_like(d_norm)
        grad_x = torch.zeros_like(d_norm)
        grad_y[:, :, :-1, :] = d_norm[:, :, 1:, :] - d_norm[:, :, :-1, :]
        grad_x[:, :, :, :-1] = d_norm[:, :, :, 1:] - d_norm[:, :, :, :-1]

        # Surface normal: [-dx, -dy, 1] normalized
        normal = torch.cat([
            -grad_x,
            -grad_y,
            torch.ones_like(d_norm)
        ], dim=1)  # (B, 3, H, W)

        # Normalize
        norm = normal.norm(dim=1, keepdim=True) + 1e-8
        normal = normal / norm

        # Map from [-1,1] to [0,1]
        normal = (normal + 1) / 2

        return F.interpolate(normal, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
