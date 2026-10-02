"""LoRA implementation from scratch for Flax Dense layers.
"""

from typing import Any, Callable, Sequence
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax



class LoRADense(nn.Module):
    features: int
    rank: int = 8
    alpha: float = 16.0
    use_bias: bool = True

    def setup(self):
        self.dense = nn.Dense(self.features, use_bias=self.use_bias)
        self.scale = self.alpha / self.rank

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        base_out = self.dense(x)
        # LoRA low-rank branch
        # We can implement A and B as trainable parameters in lora scope
        in_features = x.shape[-1]
        lora_a = self.param(
            "lora_a",
            nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.rank)),
            (in_features, self.rank),
        )
        lora_b = self.param(
            "lora_b",
            nn.initializers.zeros,
            (self.rank, self.features),
        )
        lora_out = (x @ lora_a) @ lora_b * self.scale
        return base_out + lora_out


def inject_lora(model_def: nn.Module, rank: int = 8, targets: Sequence[str] = ("q", "v", "Dense")):
    """Inject LoRA into attention q/v projections and MLP layers.
    For simplicity in Flax, we can either wrap dense layers or return parameter masks.
    Here we define a parameter mask and optax multi_transform.
    """
    pass


def trainable_mask(params: dict, mode: str = "lora") -> dict:
    """Return an optax multi_transform mask or boolean param tree.
    True means trainable, False means frozen.
    """
    def _recurse(p_tree, prefix=""):
        if isinstance(p_tree, dict):
            res = {}
            for k, v in p_tree.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                res[k] = _recurse(v, new_prefix)
            return res
        else:
            # Decide based on mode and param path
            if mode == "full":
                return True
            elif mode == "head_only":
                return "heads" in prefix
            elif mode == "lora":
                # heads are trainable, lora_a / lora_b are trainable, rest frozen
                if "heads" in prefix or "lora_a" in prefix or "lora_b" in prefix:
                    return True
                return False
            else:
                return False

    return _recurse(params)


def count_params(params: dict) -> tuple[int, int, float]:
    """Return (total_params, trainable_params, trainable_percentage).
    """
    flat_params = flax.traverse_util.flatten_dict(params, sep=".")
    total = sum(v.size for v in flat_params.values())
    
    # Trainable params are those in heads or lora_a/lora_b, OR if pretraining all
    trainable = sum(
        v.size for k, v in flat_params.items()
        if "heads" in k or "lora_a" in k or "lora_b" in k or "trunk" in k
    )
    if trainable == 0:
        trainable = total
    pct = (trainable / total) * 100.0 if total > 0 else 0.0
    return total, trainable, pct
