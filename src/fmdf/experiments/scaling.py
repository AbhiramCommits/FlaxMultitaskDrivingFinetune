"""Scaling study across 1, 2, and 4 devices measuring step time, throughput, and efficiency.
"""

import argparse
import json
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import yaml

import matplotlib.pyplot as plt
from fmdf.data.pipeline import Batcher
from fmdf.train.pretrain import DrivingModel
from fmdf.train.losses import combined_loss


def run_scaling():
    devices = jax.devices()
    device_count = len(devices)
    print(f"Available JAX devices: {device_count}")

    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    results = {}
    baseline_throughput = None

    for d in [1, 2, 4]:
        if d > device_count:
            print(f"Skipping {d} devices (only {device_count} available)")
            continue

        batcher = Batcher(
            "data/windows_train.parquet",
            batch_size=config["training"]["batch_size"],
            device_count=d,
            seq_len=config["model"]["seq_len"],
            sampler="uniform",
            is_train=True,
        )

        model = DrivingModel(
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["n_heads"],
            n_layers=config["model"]["n_layers"],
            d_ff=config["model"]["d_ff"],
            seq_len=config["model"]["seq_len"],
        )

        key = jax.random.PRNGKey(42)
        dummy = jnp.ones((d, config["training"]["batch_size"], config["model"]["seq_len"], 11))
        params = model.init(key, dummy[0], deterministic=True)["params"]

        # Replicate params across d devices
        replicated_params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * d, axis=0), params)

        def pmap_step(p, feat, le, lh, lr, w):
            def loss_fn(sub_p):
                rngs = {"dropout": jax.random.PRNGKey(42)}
                preds = model.apply({"params": sub_p}, feat, deterministic=False, rngs=rngs)
                labels = {"event": le, "horizon": lh, "rare": lr}
                return combined_loss(preds, labels, w, config["loss_weights"])

            (loss, per_task), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            grads = jax.lax.pmean(grads, axis_name="batch")
            loss = jax.lax.pmean(loss, axis_name="batch")
            return p, loss, grads

        pmap_step_fn = jax.pmap(pmap_step, axis_name="batch")

        # Warmup
        step_times = []
        for i, batch in enumerate(batcher):
            if i >= 30:
                break
            t0 = time.time()
            replicated_params, loss, grads = pmap_step_fn(
                replicated_params,
                batch.features,
                batch.labels_event,
                batch.labels_horizon,
                batch.labels_rare,
                batch.weights,
            )
            jax.block_until_ready(loss)
            dt = time.time() - t0
            if i >= 10:  # exclude 10 warmup steps
                step_times.append(dt)

        median_step_time = float(np.median(step_times))
        effective_batch = config["training"]["batch_size"] * d
        examples_per_sec = effective_batch / median_step_time

        if d == 1:
            baseline_throughput = examples_per_sec
            efficiency = 100.0
        else:
            efficiency = (examples_per_sec / (baseline_throughput * d)) * 100.0

        results[str(d)] = {
            "devices": d,
            "median_step_time_sec": median_step_time,
            "examples_per_sec": examples_per_sec,
            "scaling_efficiency_pct": efficiency,
        }
        print(f"Devices {d}: step_time={median_step_time:.4f}s, throughput={examples_per_sec:.1f} ex/s, efficiency={efficiency:.1f}%")

    os.makedirs("artifacts/results", exist_ok=True)
    os.makedirs("artifacts/figs", exist_ok=True)
    with open("artifacts/results/scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot scaling efficiency
    devs = [int(k) for k in results.keys()]
    effs = [v["scaling_efficiency_pct"] for v in results.values()]
    plt.figure(figsize=(6, 4))
    plt.plot(devs, effs, marker="o", color="b", linestyle="-")
    plt.title("Sharded Training Scaling Efficiency")
    plt.xlabel("Number of Devices")
    plt.ylabel("Efficiency (%)")
    plt.grid(True)
    plt.savefig("artifacts/figs/scaling.png")
    plt.close()
    print("Saved scaling results and figure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_scaling()
