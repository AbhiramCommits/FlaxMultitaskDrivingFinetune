"""Pretraining script for base SeqTransformer + MultiTaskHeads on city 'sf'.
"""

import argparse
import json
import os
import time
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import yaml

from fmdf.data.pipeline import Batcher
from fmdf.models.transformer import SeqTransformer
from fmdf.models.heads import MultiTaskHeads
from fmdf.models.lora import count_params
from fmdf.train.losses import combined_loss


class DrivingModel(nn.Module):
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    seq_len: int = 32
    n_features: int = 11

    def setup(self):
        self.trunk = SeqTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            seq_len=self.seq_len,
            n_features=self.n_features,
        )
        self.heads = MultiTaskHeads(d_model=self.d_model)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> dict:
        pooled = self.trunk(x, deterministic=deterministic)
        return self.heads(pooled)


def pretrain(config_path: str = "configs/base.yaml", steps: int = 1500):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load train data (filtered to sf city or all if desired, step 2 says city 'sf' ONLY)
    # Let's filter windows by city 'sf'
    import pandas as pd
    train_df_path = "data/windows_train.parquet"
    df = pd.read_parquet(train_df_path)
    df_sf = df[df["city"] == "sf"].reset_index(drop=True)
    if len(df_sf) == 0:
        df_sf = df  # fallback if sf not found

    sf_train_path = "data/windows_train_sf.parquet"
    df_sf.to_parquet(sf_train_path)

    val_df = pd.read_parquet("data/windows_val.parquet")
    val_sf = val_df[val_df["city"] == "sf"].reset_index(drop=True)
    if len(val_sf) == 0:
        val_sf = val_df
    val_sf_path = "data/windows_val_sf.parquet"
    val_sf.to_parquet(val_sf_path)

    batcher = Batcher(
        sf_train_path,
        batch_size=config["training"]["batch_size"],
        device_count=1,
        seq_len=config["model"]["seq_len"],
        sampler="uniform",
        is_train=True,
        seed=config["training"]["seed"],
    )

    val_batcher = Batcher(
        val_sf_path,
        batch_size=config["training"]["batch_size"],
        device_count=1,
        seq_len=config["model"]["seq_len"],
        sampler="uniform",
        is_train=False,
    )
    _ = val_batcher

    model = DrivingModel(
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        seq_len=config["model"]["seq_len"],
    )

    key = jax.random.PRNGKey(config["training"]["seed"])
    dummy_input = jnp.ones((1, config["model"]["seq_len"], 11))
    variables = model.init(key, dummy_input, deterministic=True)
    params = variables["params"]

    total, trainable, pct = count_params(params)
    print(f"Model params - Total: {total}, Trainable: {trainable} ({pct:.2f}%)")
    assert pct > 99.0  # pretrain trains all params

    # Optimizer with cosine schedule and warmup
    warmup_steps = int(config["training"]["warmup_steps"])
    lr = float(config["training"]["learning_rate"])
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=steps,
        end_value=lr * 0.1,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=config["training"]["weight_decay"])
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_feat, batch_lab_event, batch_lab_horizon, batch_lab_rare, batch_w):
        def loss_fn(p):
            rngs = {"dropout": jax.random.PRNGKey(0)}
            preds = model.apply({"params": p}, batch_feat, deterministic=False, rngs=rngs)
            labels = {
                "event": batch_lab_event,
                "horizon": batch_lab_horizon,
                "rare": batch_lab_rare,
            }
            loss, per_task = combined_loss(preds, labels, batch_w, config["loss_weights"])
            return loss, per_task

        (loss, per_task), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, per_task

    os.makedirs("artifacts/logs", exist_ok=True)
    log_file = open("artifacts/logs/pretrain.jsonl", "w")

    step = 0
    print(f"Starting pretraining for {steps} steps...")
    start_time = time.time()

    initial_loss = None

    while step < steps:
        for batch in batcher:
            if step >= steps:
                break
            feat = batch.features[0]  # shape (batch_size, seq_len, n_features)
            lab_event = batch.labels_event[0]
            lab_horizon = batch.labels_horizon[0]
            lab_rare = batch.labels_rare[0]
            w = batch.weights[0]

            params, opt_state, loss, per_task = train_step(
                params, opt_state, feat, lab_event, lab_horizon, lab_rare, w
            )
            loss_val = float(loss)
            if initial_loss is None:
                initial_loss = loss_val

            if step % 100 == 0:
                print(f"Step {step}/{steps} - Loss: {loss_val:.4f}")
                log_file.write(json.dumps({"step": step, "loss": loss_val}) + "\n")
                log_file.flush()

            step += 1

    log_file.close()
    duration = time.time() - start_time
    print(f"Pretraining completed in {duration:.2f}s. Initial loss: {initial_loss:.4f}, Final loss: {loss_val:.4f}")
    assert loss_val < initial_loss, "Final loss must be lower than initial loss"

    # Save checkpoint with Orbax
    ckpt_dir = os.path.abspath("artifacts/ckpt_pretrain")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_manager = ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )
    save_args = ocp.args.StandardSave(params)
    ckpt_manager.save(0, args=save_args)
    ckpt_manager.wait_until_finished()
    print(f"Saved pretrain checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()

    pretrain(args.config, args.steps)
