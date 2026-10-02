"""Interface stub and docstring for nuScenes dataset loader.

Required columns in dataframe / parquet export:
  - log_id (str or int)
  - city (str: 'sf', 'phx', 'la', etc.)
  - t (int timestep)
  - ego_vx (float)
  - ego_vy (float)
  - ego_yaw_rate (float)
  - ego_accel (float)
  - lead_dist (float)
  - lead_rel_v (float)
  - n_agents (int)
  - lane_curvature (float)
  - traffic_light_state (int)
  - weather (int)
  - time_of_day (int)
  - task_event (int: 0=cruise, 1=lane_change, 2=hard_brake, 3=yield)
  - task_horizon (float: normalized min lead dist over next H steps)
  - task_rare (int: 0 or 1, near-collision indicator)
"""

import pandas as pd


def load_nuscenes_parquet(parquet_path: str) -> pd.DataFrame:
    """Load real nuScenes / Argoverse parquet export matching the schema."""
    df = pd.read_parquet(parquet_path)
    required_cols = [
        "log_id",
        "city",
        "t",
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
        "task_event",
        "task_horizon",
        "task_rare",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required nuScenes column: {col}")
    return df
