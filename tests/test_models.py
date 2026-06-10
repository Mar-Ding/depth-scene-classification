"""Tests for CLIP cross-modal sensor adaptation."""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.sensor_adapter import MLPAdapter, CrossAttentionAdapter
from src.models.depth_processor import DepthProcessor
from src.training.loss import NTXentLoss, AlignedContrastiveLoss


def test_mlp_adapter():
    """MLPAdapter forward shape and finite output."""
    adapter = MLPAdapter(input_dim=512, hidden_dim=512, output_dim=512)
    x = torch.randn(4, 512)
    out = adapter(x)
    assert out.shape == (4, 512), f"Expected (4,512), got {out.shape}"
    assert torch.isfinite(out).all(), "Output has non-finite values"
    print("  [PASS] test_mlp_adapter")


def test_cross_attention_adapter():
    """CrossAttentionAdapter forward shapes."""
    adapter = CrossAttentionAdapter(
        input_dim=512, hidden_dim=512, output_dim=512,
        num_queries=16, num_heads=8
    )
    x = torch.randn(4, 50, 512)
    global_emb, local_tokens, attn_weights = adapter(x)
    assert global_emb.shape == (4, 512), f"Expected (4,512), got {global_emb.shape}"
    assert local_tokens.shape == (4, 16, 512), f"Expected (4,16,512), got {local_tokens.shape}"
    assert attn_weights.shape == (4, 8, 16, 50), f"Expected (4,8,16,50), got {attn_weights.shape}"
    print("  [PASS] test_cross_attention_adapter")


def test_ntxent_loss():
    """NT-Xent loss values are finite and positive."""
    loss_fn = NTXentLoss(temperature=0.07)
    z1 = torch.randn(8, 512)
    z2 = torch.randn(8, 512)
    loss = loss_fn(z1, z2)
    assert torch.isfinite(loss), "Loss is not finite"
    assert loss > 0, f"Loss should be positive, got {loss}"
    loss_perfect = loss_fn(z1, z1)
    assert loss_perfect < loss, "Perfect alignment should have lower loss"
    print(f"  [PASS] test_ntxent_loss (loss={loss.item():.4f}, perfect={loss_perfect.item():.4f})")


def test_aligned_contrastive_loss():
    """AlignedContrastiveLoss integrates depth and RGB features."""
    loss_fn = AlignedContrastiveLoss(temperature=0.07)
    depth_embeds = torch.randn(8, 512)
    rgb_embeds = torch.randn(8, 512)
    loss = loss_fn(depth_embeds, rgb_embeds)
    assert torch.isfinite(loss), "Loss is not finite"
    print(f"  [PASS] test_aligned_contrastive_loss (loss={loss.item():.4f})")


def test_depth_processor():
    """DepthProcessor produces correct output shapes for all strategies."""
    for strategy in ["repeat", "normal"]:
        dp = DepthProcessor(strategy=strategy, image_size=224)
        depth = torch.randn(2, 1, 480, 640)
        out = dp(depth)
        assert out.shape == (2, 3, 224, 224), \
            f"[{strategy}] Expected (2,3,224,224), got {out.shape}"
        assert out.min() >= 0 and out.max() <= 1, \
            f"[{strategy}] Output not in [0,1]"
    print("  [PASS] test_depth_processor")


def test_adapter_training_step():
    """Full training step: adapter forward through contrastive loss."""
    adapter = MLPAdapter(input_dim=512, hidden_dim=512, output_dim=512)
    loss_fn = AlignedContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    depth_feats = torch.randn(8, 512)
    rgb_feats = torch.randn(8, 512)

    adapted = adapter(depth_feats)
    loss1 = loss_fn(adapted, rgb_feats)

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    adapted2 = adapter(depth_feats)
    loss2 = loss_fn(adapted2, rgb_feats)

    print(f"  [PASS] test_adapter_training_step (loss {loss1.item():.4f} -> {loss2.item():.4f})")


if __name__ == "__main__":
    print("Running adapter & loss tests...")
    test_mlp_adapter()
    test_cross_attention_adapter()
    test_ntxent_loss()
    test_aligned_contrastive_loss()
    test_depth_processor()
    test_adapter_training_step()
    print("\nAll tests passed! (CLIP-dependent tests skipped)")
