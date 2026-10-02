"""Tests for DuckDB SQL assembly and deterministic splitting.
"""

import duckdb
import pandas as pd


def test_sql_windowing_fixture():
    con = duckdb.connect(database=":memory:")
    _raw_df = pd.DataFrame({
        "log_id": ["log_1"] * 40,
        "city": ["sf"] * 40,
        "t": list(range(40)),
        "ego_vx": [10.0] * 40,
        "ego_vy": [0.0] * 40,
        "ego_yaw_rate": [0.0] * 40,
        "ego_accel": [0.0] * 40,
        "lead_dist": [20.0] * 40,
        "lead_rel_v": [0.0] * 40,
        "n_agents": [2] * 40,
        "lane_curvature": [0.0] * 40,
        "traffic_light_state": [0] * 40,
        "weather": [0] * 40,
        "time_of_day": [0] * 40,
        "task_event": [0] * 40,
        "task_horizon": [0.5] * 40,
        "task_rare": [0] * 40,
    })
    con.execute("CREATE TABLE raw AS SELECT * FROM _raw_df")
    with open("queries/assemble_windows.sql", "r") as f:
        con.execute(f.read())
    windows = con.execute("SELECT * FROM windows").fetchdf()
    assert len(windows) > 0, "SQL windowing should produce windows"
    con.close()
