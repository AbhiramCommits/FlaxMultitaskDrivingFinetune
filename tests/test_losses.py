"""Tests for multi-task loss functions and importance weighting.
"""

import jax.numpy as jnp
from fmdf.train.losses import combined_loss


def test_combined_loss_properties():
    preds = {
        "event": jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "horizon": jnp.array([0.5, 0.2]),
        "rare": jnp.array([0.1, -0.5]),
    }
    labels = {
        "event": jnp.array([0, 1]),
        "horizon": jnp.array([0.5, 0.2]),
        "rare": jnp.array([0, 1]),
    }
    weights = jnp.array([1.0, 1.0])
    loss_weights = {"event": 1.0, "horizon": 1.0, "rare": 1.0, "rare_pos_weight": 1.0}

    total_loss, per_task = combined_loss(preds, labels, weights, loss_weights)
    assert total_loss.ndim == 0, "Loss must be scalar"
    assert "event" in per_task
    assert "horizon" in per_task
    assert "rare" in per_task
