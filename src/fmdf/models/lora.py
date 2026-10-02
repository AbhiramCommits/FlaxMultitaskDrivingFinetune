"""LoRA injection and parameter masking for fine-tuning.
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


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


def inject_lora_params(params: dict, rank: int = 8) -> dict:
    """Inject LoRA parameters into attention projection layers or MLP layers selectively to keep trainable share < 5%.
    """
    def _inject(subdict, prefix=""):
        if not isinstance(subdict, dict):
            return subdict
        # Inject only in self_attention query/value or specific dense layers
        is_target = "self_attention" in prefix and ("query" in prefix or "value" in prefix)
        if "kernel" in subdict and hasattr(subdict["kernel"], "shape") and len(subdict["kernel"].shape) == 2 and is_target:
            res = {}
            for k, v in subdict.items():
                res[k] = _inject(v, f"{prefix}.{k}" if prefix else k)
            v_kernel = subdict["kernel"]
            in_dim, out_dim = v_kernel.shape
            key = jax.random.PRNGKey(42)
            res["lora_a"] = jax.random.normal(key, (in_dim, rank)) * (1.0 / jnp.sqrt(rank))
            res["lora_b"] = jnp.zeros((rank, out_dim))
            return res
        else:
            res = {}
            for k, v in subdict.items():
                res[k] = _inject(v, f"{prefix}.{k}" if prefix else k)
            return res

    return _inject(params)


def get_trainable_mask(params: dict, mode: str = "lora") -> dict:
    """Return boolean pytree for optax.masked or gradient masking.
    """
    flat = flax.traverse_util.flatten_dict(params, sep=".")
    mask = {}
    for k, v in flat.items():
        if mode == "full":
            mask[k] = True
        elif mode == "head_only":
            mask[k] = "heads" in k
        elif mode == "lora":
            mask[k] = "heads" in k or "lora_a" in k or "lora_b" in k
        else:
            mask[k] = False
    return mask


def count_params_lora(params: dict, mode: str = "lora") -> tuple[int, int, float]:
    flat = flax.traverse_util.flatten_dict(params, sep=".")
    total = sum(v.size for v in flat.values())
    trainable = 0
    for k, v in flat.items():
        if mode == "full":
            is_tr = True
        elif mode == "head_only":
            is_tr = "heads" in k
        elif mode == "lora":
            is_tr = "heads" in k or "lora_a" in k or "lora_b" in k
        else:
            is_tr = False
        if is_tr:
            trainable += v.size
    pct = (trainable / total) * 100.0 if total > 0 else 0.0
    return total, trainable, pct


def count_params(params: dict, mode: str = "lora") -> tuple[int, int, float]:
    return count_params_lora(params, mode)
