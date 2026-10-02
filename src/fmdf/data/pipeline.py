"""Input pipeline with normalization, importance weighting, and sharded batching.
"""

import json
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Batch:
    features: np.ndarray  # shape: (device_count, per_device_batch, seq_len, n_features)
    labels_event: np.ndarray
    labels_horizon: np.ndarray
    labels_rare: np.ndarray
    weights: np.ndarray


class Batcher:
    def __init__(
        self,
        parquet_path: str,
        batch_size: int = 32,
        device_count: int = 4,
        seq_len: int = 32,
        sampler: str = "uniform",
        norm_stats_path: str = "artifacts/norm_stats.json",
        is_train: bool = True,
        seed: int = 42,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.batch_size = batch_size
        self.device_count = device_count
        self.seq_len = seq_len
        self.sampler = sampler
        self.is_train = is_train
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.feature_cols = [
            "ego_vx",
            "ego_vy",
            "ego_yaw_rate",
            "ego_accel",
            "lead_dist",
            "lead_rel_v",
            "n_agents",
            "lane_curvature",
            "traffic_light_state",
            "weather",
            "time_of_day",
        ]

        # Compute or load norm stats (TRAIN-ONLY)
        os.makedirs(os.path.dirname(norm_stats_path), exist_ok=True)
        if is_train and not os.path.exists(norm_stats_path):
            self.means = {}
            self.stds = {}
            for col in self.feature_cols:
                all_vals = np.concatenate(self.df[col + "_seq"].values)
                m = float(np.mean(all_vals))
                s = float(np.std(all_vals))
                self.means[col] = m
                self.stds[col] = max(s, 1e-5)
            with open(norm_stats_path, "w") as f:
                json.dump({"means": self.means, "stds": self.stds}, f)
        else:
            if os.path.exists(norm_stats_path):
                with open(norm_stats_path, "r") as f:
                    stats = json.load(f)
                    self.means = stats["means"]
                    self.stds = stats["stds"]
            else:
                self.means = {col: 0.0 for col in self.feature_cols}
                self.stds = {col: 1.0 for col in self.feature_cols}

        self.n_samples = len(self.df)
        self.effective_batch_size = batch_size * device_count
        self.steps_per_epoch = max(1, self.n_samples // self.effective_batch_size)

        # Importance weights for rare task
        rare_labels = self.df["task_rare"].values
        if sampler == "importance":
            pos_rate = max(1.0 - rare_labels.mean(), 0.01)
            # Higher weight for rare class (1)
            self.sample_weights = np.where(rare_labels == 1, 1.0 / pos_rate, 1.0 / (1.0 - pos_rate))
            self.sampling_probs = np.where(rare_labels == 1, 0.5, 0.5 / (self.n_samples - rare_labels.sum()))
            self.sampling_probs /= self.sampling_probs.sum()
        else:
            self.sample_weights = np.ones(self.n_samples)
            self.sampling_probs = None

    def __iter__(self):
        if self.is_train and self.sampling_probs is not None:
            indices = self.rng.choice(self.n_samples, size=self.n_samples, p=self.sampling_probs, replace=True)
        else:
            indices = np.arange(self.n_samples)
            if self.is_train:
                self.rng.shuffle(indices)

        # Trim to multiple of effective_batch_size
        n_usable = (self.n_samples // self.effective_batch_size) * self.effective_batch_size
        indices = indices[:n_usable]

        for start_idx in range(0, n_usable, self.effective_batch_size):
            batch_indices = indices[start_idx : start_idx + self.effective_batch_size]
            batch_df = self.df.iloc[batch_indices]

            # Construct feature array: (effective_batch_size, seq_len, n_features)
            feat_list = []
            for _, row in batch_df.iterrows():
                seq_mat = np.stack([
                    (np.array(row[col + "_seq"][: self.seq_len]) - self.means[col]) / self.stds[col]
                    for col in self.feature_cols
                ], axis=-1)
                # Ensure exact seq_len by padding if necessary
                if len(seq_mat) < self.seq_len:
                    pad = np.zeros((self.seq_len - len(seq_mat), len(self.feature_cols)))
                    seq_mat = np.concatenate([seq_mat, pad], axis=0)
                feat_list.append(seq_mat)

            features = np.stack(feat_list, axis=0)
            # Reshape to (device_count, per_device_batch, seq_len, n_features)
            per_dev_batch = self.batch_size
            features = features.reshape(self.device_count, per_dev_batch, self.seq_len, len(self.feature_cols))

            labels_event = batch_df["task_event"].values.reshape(self.device_count, per_dev_batch)
            labels_horizon = batch_df["task_horizon"].values.reshape(self.device_count, per_dev_batch)
            labels_rare = batch_df["task_rare"].values.reshape(self.device_count, per_dev_batch)
            weights = self.sample_weights[batch_indices].reshape(self.device_count, per_dev_batch)

            yield Batch(
                features=features,
                labels_event=labels_event,
                labels_horizon=labels_horizon,
                labels_rare=labels_rare,
                weights=weights,
            )
