"""Shared pytest fixtures and test environment setup.
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Generate tiny fixture dataset if not exists
    if not os.path.exists("data/windows_train.parquet"):
        from fmdf.data.generate import generate_logs
        from fmdf.data.sql import build_dataset
        df = generate_logs(n_logs=50, seed=42)
        df.to_parquet("data/raw.parquet")
        build_dataset("data/raw.parquet", "data")
