"""Tests for LoRA parameter share and freezing behavior.
"""

import jax
import jax.numpy as jnp
from fmdf.train.pretrain import DrivingModel
from fmdf.models.lora import inject_lora_params, count_params_lora


def test_lora_share_and_freezing():
    model = DrivingModel(d_model=64, n_heads=2, n_layers=2, d_ff=128, seq_len=32)
    key = jax.random.PRNGKey(42)
    x = jnp.ones((1, 32, 11))
    params = model.init(key, x, deterministic=True)["params"]

    lora_params = inject_lora_params(params, rank=4)
    total, trainable, pct = count_params_lora(lora_params, mode="lora")

    assert pct < 5.0, f"LoRA share {pct}% must be < 5%"
    assert trainable > 0, "Trainable params must be > 0"
    assert total > trainable, "Total params must exceed trainable params"
