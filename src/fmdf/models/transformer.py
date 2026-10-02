"""Flax linen transformer trunk for driving sequence modeling.
"""

from typing import Any
import flax.linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Pre-LN Transformer Block
        # Self-attention
        norm_x = nn.LayerNorm()(x)
        attn = nn.SelfAttention(
            num_heads=self.n_heads,
            dtype=x.dtype,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout,
        )(norm_x, deterministic=deterministic)
        x = x + attn

        # MLP
        norm_x2 = nn.LayerNorm()(x)
        mlp = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.relu,
            nn.Dropout(self.dropout, deterministic=deterministic),
            nn.Dense(self.d_model),
            nn.Dropout(self.dropout, deterministic=deterministic),
        ])(norm_x2)
        x = x + mlp
        return x


class SeqTransformer(nn.Module):
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    seq_len: int = 32
    n_features: int = 11

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x shape: (batch, seq_len, n_features)
        bsz, seq_len, _ = x.shape
        h = nn.Dense(self.d_model, name="input_proj")(x)

        # Learned positional embeddings
        pos_emb = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.seq_len, self.d_model),
        )
        h = h + pos_emb[:, :seq_len, :]
        h = nn.Dropout(self.dropout, deterministic=deterministic)(h)

        for i in range(self.n_layers):
            h = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                name=f"block_{i}",
            )(h, deterministic=deterministic)

        h = nn.LayerNorm(name="final_norm")(h)
        # Pooled representation: mean pool + final timestep concat or mean pool
        pooled = jnp.mean(h, axis=1)  # (batch, d_model)
        return pooled
