"""Fine-tuning script supporting LoRA, Full FT, and Head-Only modes at equal compute.
"""

import argparse
import json
import os
import time
import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import yaml

from fmdf.data.pipeline import Batcher
from fmdf.train.pretrain import DrivingModel
from fmdf.train.losses import combined_loss
from fmdf.models.lora import inject_lora_params, get_trainable_mask, count_params_lora


import numpy as np


def compute_metrics(preds_list, labels_list):
    # Quick eval metrics for validation reporting
    from sklearn.metrics import f1_score, mean_absolute_error, average_precision_score
    events = np.concatenate([p["event"] for p in preds_list])
    event_labs = np.concatenate([lbl["event"] for lbl in labels_list])
    macro_f1 = float(f1_score(event_labs, np.argmax(events, axis=-1), average="macro"))

    horizons = np.concatenate([p["horizon"] for p in preds_list])
    horizon_labs = np.concatenate([lbl["horizon"] for lbl in labels_list])
    mae = float(mean_absolute_error(horizon_labs, horizons))

    rares = np.concatenate([p["rare"] for p in preds_list])
    rare_labs = np.concatenate([lbl["rare"] for lbl in labels_list])
    rare_probs = 1.0 / (1.0 + np.exp(-rares))
    try:
        pr_auc = float(average_precision_score(rare_labs, rare_probs))
    except Exception:
        pr_auc = 0.0

    return {"event_macro_f1": macro_f1, "horizon_mae": mae, "rare_pr_auc": pr_auc}


def finetune(mode: str = "lora", config_path: str = "configs/base.yaml", steps: int = 500):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    batcher = Batcher(
        "data/windows_train.parquet",
        batch_size=config["training"]["batch_size"],
        device_count=1,
        seq_len=config["model"]["seq_len"],
        sampler="importance",
        is_train=True,
        seed=config["training"]["seed"],
    )

    val_batcher = Batcher(
        "data/windows_val.parquet",
        batch_size=config["training"]["batch_size"],
        device_count=1,
        seq_len=config["model"]["seq_len"],
        sampler="uniform",
        is_train=False,
    )

    model = DrivingModel(
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        seq_len=config["model"]["seq_len"],
    )

    # Restore pretrain checkpoint
    ckpt_dir = os.path.abspath("artifacts/ckpt_pretrain")
    ckpt_manager = ocp.CheckpointManager(ckpt_dir)
    target_step = ckpt_manager.latest_step()
    if target_step is None:
        target_step = 0
    restored_params = ckpt_manager.restore(target_step)
    if isinstance(restored_params, tuple):
        params = restored_params[0]
    elif isinstance(restored_params, dict) and "params" in restored_params:
        params = restored_params["params"]
    else:
        params = restored_params

    if mode == "lora":
        params = inject_lora_params(params, rank=config["lora"]["rank"])

    total, trainable, pct = count_params_lora(params, mode)
    print(f"Mode: {mode} | Total params: {total}, Trainable: {trainable} ({pct:.2f}%)")
    if mode == "lora":
        assert pct < 5.0, f"LoRA trainable share {pct}% must be < 5%"

    lr = float(config["training"]["learning_rate"])
    optimizer = optax.adamw(learning_rate=lr, weight_decay=config["training"]["weight_decay"])
    opt_state = optimizer.init(params)

    # Custom gradient update wrapper instead of optax.masked to avoid pytree metadata prefix issues
    @jax.jit
    def train_step(params, opt_state, batch_feat, batch_lab_event, batch_lab_horizon, batch_lab_rare, batch_w):
        def loss_fn(p):
            rngs = {"dropout": jax.random.PRNGKey(42)}
            preds = model.apply({"params": p}, batch_feat, deterministic=False, rngs=rngs)
            labels = {
                "event": batch_lab_event,
                "horizon": batch_lab_horizon,
                "rare": batch_lab_rare,
            }
            loss, per_task = combined_loss(preds, labels, batch_w, config["loss_weights"])
            return loss, per_task

        (loss, per_task), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Zero out gradients for frozen parameters based on mode
        flat_grads = flax.traverse_util.flatten_dict(grads, sep=".")
        flat_mask = get_trainable_mask(params, mode)
        
        new_flat_grads = {}
        for k, g in flat_grads.items():
            if flat_mask.get(k, False):
                new_flat_grads[k] = g
            else:
                new_flat_grads[k] = jnp.zeros_like(g)
        grads = flax.traverse_util.unflatten_dict(new_flat_grads, sep=".")

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, per_task

    step = 0
    start_time = time.time()
    print(f"Starting fine-tuning ({mode}) for {steps} steps (equal-compute protocol)...")

    while step < steps:
        for batch in batcher:
            if step >= steps:
                break
            feat = batch.features[0]
            lab_event = batch.labels_event[0]
            lab_horizon = batch.labels_horizon[0]
            lab_rare = batch.labels_rare[0]
            w = batch.weights[0]

            params, opt_state, loss, per_task = train_step(
                params, opt_state, feat, lab_event, lab_horizon, lab_rare, w
            )
            step += 1
            if step % 100 == 0:
                print(f"Fine-tune {mode} step {step}/{steps} - Loss: {float(loss):.4f}")

    duration = time.time() - start_time
    steps_per_sec = steps / duration

    # Eval on val set
    preds_list = []
    labels_list = []
    for batch in val_batcher:
        feat = batch.features[0]
        preds = model.apply({"params": params}, feat, deterministic=True)
        preds_list.append({k: np.array(v) for k, v in preds.items()})
        labels_list.append({
            "event": batch.labels_event[0],
            "horizon": batch.labels_horizon[0],
            "rare": batch.labels_rare[0],
        })

    metrics = compute_metrics(preds_list, labels_list)
    result = {
        "mode": mode,
        "wall_clock_seconds": duration,
        "steps_per_sec": steps_per_sec,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percentage": pct,
        "metrics": metrics,
    }

    os.makedirs("artifacts/results", exist_ok=True)
    out_json = f"artifacts/results/finetune_{mode}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {mode} results to {out_json}: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="lora", choices=["lora", "full", "head_only"])
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    finetune(args.mode, args.config, args.steps)
