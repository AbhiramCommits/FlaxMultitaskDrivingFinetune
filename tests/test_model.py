"""Tests for model forward pass outputs and determinism.
"""

import jax
import jax.numpy as jnp
from fmdf.train.pretrain import DrivingModel


def test_model_forward_shapes_and_determinism():
    model = DrivingModel(d_model=64, n_heads=2, n_layers=2, d_ff=128, seq_len=32)
    key = jax.random.PRNGKey(42)
    x = jnp.ones((2, 32, 11))
    params = model.init(key, x, deterministic=True)["params"]

    preds1 = model.apply({"params": params}, x, deterministic=True)
    preds2 = model.apply({"params": params}, x, deterministic=True)

    assert preds1["event"].shape == (2, 4)
    assert preds1["horizon"].shape == (2,)
    assert preds1["rare"].shape == (2,)

    # Check determinism
    assert jnp.allclose(preds1["event"], preds2["event"])
    assert jnp.allclose(preds1["horizon"], preds2["horizon"])
    assert jnp.allclose(preds1["rare"], preds2["rare"])
