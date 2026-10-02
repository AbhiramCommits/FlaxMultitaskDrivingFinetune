"""Multi-task loss functions with importance weighting and focal-style components.
"""

import jax
import jax.numpy as jnp


def combined_loss(
    preds: dict,
    labels: dict,
    weights: jnp.ndarray,
    loss_weights: dict,
) -> tuple[jnp.ndarray, dict]:
    """Compute combined multi-task loss with per-example importance weights.
    
    preds: dict with keys 'event', 'horizon', 'rare'
    labels: dict with keys 'event', 'horizon', 'rare'
    weights: per-example importance weights (batch_size,)
    loss_weights: dict with weights for each task
    """
    # 1. Event loss: softmax cross entropy (multiclass 4 classes)
    event_logits = preds["event"]  # (batch, 4)
    event_labels = labels["event"]  # (batch,)
    one_hot_event = jax.nn.one_hot(event_labels, 4)
    event_ce = -jnp.sum(one_hot_event * jax.nn.log_softmax(event_logits), axis=-1)
    event_loss = jnp.mean(event_ce * weights)

    # 2. Horizon loss: Huber loss
    horizon_pred = preds["horizon"]  # (batch,)
    horizon_labels = labels["horizon"]  # (batch,)
    diff = horizon_pred - horizon_labels
    delta = 1.0
    huber = jnp.where(jnp.abs(diff) < delta, 0.5 * diff ** 2, delta * (jnp.abs(diff) - 0.5 * delta))
    horizon_loss = jnp.mean(huber * weights)

    # 3. Rare loss: weighted binary cross entropy with focal/pos weight
    rare_logit = preds["rare"]  # (batch,)
    rare_labels = labels["rare"]  # (batch,)
    pos_weight = loss_weights.get("rare_pos_weight", 10.0)
    
    # BCE with logits and pos_weight
    max_val = jnp.maximum(0, rare_logit)
    bce = max_val - rare_logit * rare_labels + jnp.log(1 + jnp.exp(-jnp.abs(rare_logit)))
    # apply pos_weight to positive class
    bce_weighted = bce * jnp.where(rare_labels == 1, pos_weight, 1.0)
    rare_loss = jnp.mean(bce_weighted * weights)

    # Total combined loss
    w_event = loss_weights.get("event", 1.0)
    w_horizon = loss_weights.get("horizon", 1.0)
    w_rare = loss_weights.get("rare", 5.0)

    total = w_event * event_loss + w_horizon * horizon_loss + w_rare * rare_loss

    per_task = {
        "event": event_loss,
        "horizon": horizon_loss,
        "rare": rare_loss,
    }
    return total, per_task
