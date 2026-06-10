"""Quick test without CLIP dependency — verifies training loop logic."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.models.sensor_adapter import MLPAdapter
from src.training.loss import AlignedContrastiveLoss


def mock_training():
    """Test full training loop with mock CLIP features."""
    print("=" * 50)
    print("Mock Training Test (synthetic data, no CLIP)")
    print("=" * 50)

    cfg = Config(
        adapter_type="mlp",
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-3,
        adapter_hidden_dim=512,
    )
    print(f"Device: {cfg.device}")

    adapter = MLPAdapter(
        input_dim=512, hidden_dim=512, output_dim=512,
        num_layers=2, dropout=0.1
    ).to(cfg.device)
    print(f"Adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    loss_fn = AlignedContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    n_batches = 100
    B = cfg.batch_size
    history = {"loss": []}

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            depth_feats = torch.randn(B, 512, device=cfg.device)
            rgb_feats = torch.randn(B, 512, device=cfg.device)

            adapted = adapter(depth_feats)
            loss = loss_fn(adapted, rgb_feats)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        converged = avg_loss < history["loss"][0] - 0.1
        status = "OK converging" if converged else "..."
        print(f"  Epoch {epoch+1:2d}/{cfg.num_epochs} | Loss: {avg_loss:.4f} {status}")

    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    assert final_loss < initial_loss, \
        f"Training did not converge: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"\n  Convergence verified: {initial_loss:.4f} -> {final_loss:.4f}")
    print("\n" + "=" * 50)
    print("Mock training test PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    mock_training()
