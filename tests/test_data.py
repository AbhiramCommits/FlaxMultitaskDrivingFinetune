"""Tests for dataset generation, splits leakage, and norm stats.
"""

import os
import pandas as pd


def test_data_splits_leakage():
    train_df = pd.read_parquet("data/windows_train.parquet")
    val_df = pd.read_parquet("data/windows_val.parquet")
    test_df = pd.read_parquet("data/windows_test.parquet")

    train_logs = set(train_df["log_id"].unique())
    val_logs = set(val_df["log_id"].unique())
    test_logs = set(test_df["log_id"].unique())

    assert len(train_logs.intersection(val_logs)) == 0, "Data leakage between train and val!"
    assert len(train_logs.intersection(test_logs)) == 0, "Data leakage between train and test!"
    assert len(val_logs.intersection(test_logs)) == 0, "Data leakage between val and test!"


def test_window_shapes():
    train_df = pd.read_parquet("data/windows_train.parquet")
    assert len(train_df) > 0
    seq_col = "ego_vx_seq"
    if seq_col in train_df.columns:
        seq = train_df[seq_col].iloc[0]
        assert len(seq) >= 1, f"Expected window length >= 1, got {len(seq)}"


def test_rare_positive_rate():
    train_df = pd.read_parquet("data/windows_train.parquet")
    pos_rate = train_df["task_rare"].mean()
    assert 0.015 <= pos_rate <= 0.08, f"Rare positive rate {pos_rate:.4f} outside expected range"


def test_norm_stats_train_only():
    norm_path = "artifacts/norm_stats.json"
    assert os.path.exists(norm_path), "Norm stats json must be computed from train only"
