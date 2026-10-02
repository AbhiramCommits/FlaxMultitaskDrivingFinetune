"""Multi-task heads module for driving model.
"""

import flax.linen as nn
import jax.numpy as jnp


class MultiTaskHeads(nn.Module):
    d_model: int = 128

    @nn.compact
    def __call__(self, pooled: jnp.ndarray) -> dict:
        # Event head: multiclass 4 classes
        event_logits = nn.Dense(4, name="event_head")(pooled)

        # Horizon head: regression 1 value (scalar)
        horizon_pred = nn.Dense(1, name="horizon_head")(pooled)

        # Rare head: binary logit
        rare_logit = nn.Dense(1, name="rare_head")(pooled)

        return {
            "event": event_logits,
            "horizon": horizon_pred.squeeze(-1),
            "rare": rare_logit.squeeze(-1),
        }
