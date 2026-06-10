"""Sensor Adapter models for cross-modal alignment."""

import torch
import torch.nn as nn
import math


class MLPAdapter(nn.Module):
    """Simple MLP Adapter (T-CLIP style).

    Maps depth features from CLIP vision encoder to aligned feature space.
    Architecture: Linear → LN → ReLU → Dropout → Linear → LN
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512,
                 output_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.extend([
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])

        # Remove last ReLU + Dropout, keep last LayerNorm
        layers = layers[:-2]  # remove last dropout + relu
        # Add final layer normalization
        layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, D) input features from CLIP vision encoder

        Returns:
            (B, D) aligned features
        """
        return self.mlp(x)


class CrossAttentionAdapter(nn.Module):
    """Cross-Attention Adapter (proposal's main design).

    Uses learnable query tokens to attend to sensor patch tokens,
    producing both global and local representations.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512,
                 output_dim: int = 512, num_queries: int = 16,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_queries = num_queries

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

        # Token projection for local tokens
        self.token_proj = nn.Linear(input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, patch_tokens: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            patch_tokens: (B, N, D) sensor patch tokens from CLIP vision encoder

        Returns:
            global_embedding: (B, D) global representation for contrastive learning
            local_tokens: (B, M, D) refined tokens for local alignment
        """
        B, N, D = patch_tokens.shape

        # Project patch tokens to hidden dim
        t_proj = self.token_proj(patch_tokens)  # (B, N, H)

        # Expand query tokens to batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, M, H)

        # Cross-attention: queries attend to patch tokens
        attn_out, attn_weights = self.cross_attn(
            queries, t_proj, t_proj,
            need_weights=True,
            average_attn_weights=False
        )
        queries = self.layer_norm1(queries + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(queries)
        queries = self.layer_norm2(queries + self.dropout(ffn_out))

        # Global representation: mean pool queries
        global_emb = queries.mean(dim=1)  # (B, H)
        global_emb = self.output_proj(global_emb)
        global_emb = self.output_norm(global_emb)

        # Normalize
        global_emb = global_emb / (global_emb.norm(dim=-1, keepdim=True) + 1e-8)

        # Local tokens = queries (for local alignment loss)
        local_tokens = self.output_proj(queries.view(-1, self.hidden_dim))
        local_tokens = local_tokens.view(B, self.num_queries, -1)

        return global_emb, local_tokens, attn_weights


def create_adapter(adapter_type: str, input_dim: int = 512, **kwargs) -> nn.Module:
    """Factory function for adapter creation."""
    if adapter_type == "mlp":
        return MLPAdapter(input_dim=input_dim, **kwargs)
    elif adapter_type == "cross_attn":
        return CrossAttentionAdapter(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
