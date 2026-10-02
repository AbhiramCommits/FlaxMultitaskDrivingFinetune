"""Tests for pmap sharding equivalence.
"""

import jax
import jax.numpy as jnp
from fmdf.train.pretrain import DrivingModel


def test_sharding_equivalence():
    devices = jax.devices()
    if len(devices) < 2:
        return  # skip if < 2 devices

    model = DrivingModel(d_model=64, n_heads=2, n_layers=2, d_ff=128, seq_len=32)
    key = jax.random.PRNGKey(42)
    x = jnp.ones((2, 16, 32, 11))
    dummy_input = jnp.ones((1, 32, 11))
    params = model.init(key, dummy_input, deterministic=True)["params"]

    replicated_params = jax.tree_util.tree_map(lambda arr: jnp.stack([arr, arr]), params)

    @jax.pmap
    def step_fn(p, batch_feat):
        preds = model.apply({"params": p}, batch_feat, deterministic=True)
        return jnp.mean(preds["horizon"])

    res = step_fn(replicated_params, x)
    assert res.shape == (2,), f"Expected pmap output shape (2,), got {res.shape}"
    assert jnp.allclose(res[0], res[1], atol=1e-5)
