"""Deterministic synthetic driving-log generator mimicking nuScenes schema and statistics.
"""

import argparse
import numpy as np
import pandas as pd


def generate_logs(n_logs: int = 1200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    cities = ["sf", "phx", "la"]
    city_weights = [0.4, 0.3, 0.3]

    city_params = {
        "sf": {
            "speed_mean": 10.0,
            "speed_std": 3.0,
            "curv_scale": 0.05,
            "agent_density": 5,
            "brake_prob": 0.05,
        },
        "phx": {
            "speed_mean": 15.0,
            "speed_std": 4.0,
            "curv_scale": 0.01,
            "agent_density": 3,
            "brake_prob": 0.03,
        },
        "la": {
            "speed_mean": 12.0,
            "speed_std": 5.0,
            "curv_scale": 0.03,
            "agent_density": 8,
            "brake_prob": 0.08,
        },
    }

    records = []
    for log_idx in range(n_logs):
        city = np.random.choice(cities, p=city_weights)
        params = city_params[city]
        seq_len = np.random.randint(40, 80)

        # Autoregressive kinematic rollout
        vx = np.clip(np.random.normal(params["speed_mean"], params["speed_std"]), 2.0, 30.0)
        vy = np.random.normal(0.0, 0.5, size=seq_len)
        yaw_rate = np.random.normal(0.0, params["curv_scale"], size=seq_len)
        accel = np.random.normal(0.0, 1.0, size=seq_len)
        lead_dist = np.random.uniform(15.0, 50.0, size=seq_len)
        lead_rel_v = np.random.normal(0.0, 2.0, size=seq_len)

        log_id = f"log_{city}_{log_idx:04d}"

        # Simulate rollout
        for t in range(seq_len):
            # update vx with autocorrelation
            if t > 0:
                vx = np.clip(vx + np.random.normal(0.0, 0.5), 1.0, 35.0)
                lead_dist[t] = np.clip(
                    lead_dist[t - 1] + lead_rel_v[t - 1] * 0.1 + np.random.normal(0.0, 0.5),
                    2.0,
                    100.0,
                )
                lead_rel_v[t] = np.clip(
                    lead_rel_v[t - 1] + np.random.normal(0.0, 0.5), -15.0, 15.0
                )

            # Event label derivation
            # 0: cruise, 1: lane_change, 2: hard_brake, 3: yield
            yr = abs(yaw_rate[t])
            acc = accel[t]
            ld = lead_dist[t]

            if acc < -2.5:
                event = 2  # hard_brake
            elif yr > 0.03:
                event = 1  # lane_change
            elif ld < 10.0 or lead_rel_v[t] < -3.0:
                event = 3  # yield
            else:
                event = 0  # cruise

            records.append(
                {
                    "log_id": log_id,
                    "city": city,
                    "t": t,
                    "ego_vx": float(vx),
                    "ego_vy": float(vy[t]),
                    "ego_yaw_rate": float(yaw_rate[t]),
                    "ego_accel": float(accel[t]),
                    "lead_dist": float(lead_dist[t]),
                    "lead_rel_v": float(lead_rel_v[t]),
                    "n_agents": int(
                        np.clip(
                            np.random.poisson(params["agent_density"]), 0, 20
                        )
                    ),
                    "lane_curvature": float(yaw_rate[t] * params["curv_scale"]),
                    "traffic_light_state": int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])),
                    "weather": int(np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])),
                    "time_of_day": int(np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])),
                    "task_event": int(event),
                }
            )

    df = pd.DataFrame(records)

    # Compute task_horizon (min lead_dist over next 10 steps, normalized)
    # Group by log_id and compute rolling min over 10 steps ahead
    horizons = []
    for _, group in df.groupby("log_id"):
        ld_vals = group["lead_dist"].values
        h_vals = []
        n = len(ld_vals)
        for i in range(n):
            end_idx = min(n, i + 10)
            min_ld = np.min(ld_vals[i:end_idx])
            h_vals.append(min_ld)
        horizons.extend(h_vals)

    df["task_horizon"] = np.array(horizons) / 100.0  # normalize to [0, 1] roughly

    # Compute task_rare: near-collision when horizon TTC < threshold
    # TTC proxy: lead_dist / max(1e-3, -lead_rel_v) when rel_v < 0, else large.
    # Alternatively, threshold on lead_dist < 8.0 and accel < -2.0 to hit 1.5%-3%
    rare_scores = (df["lead_dist"] < 9.0) & (df["ego_accel"] < -2.0)
    # Adjust positive rate to be strictly between 1.5% and 3.0%
    # If too high or low, tune threshold or randomly sample
    n_total = len(df)
    target_pos_rate = 0.022  # 2.2%
    current_pos = rare_scores.sum() / n_total

    # Let's make it deterministic & tunable via condition
    rare_mask = (df["lead_dist"] < 8.5) & (df["ego_accel"] < -1.8) | (
        (df["lead_dist"] < 6.0) & (df["lead_rel_v"] < -4.0)
    )
    # Ensure exact rate by fine adjustment if needed, but let's check
    df["task_rare"] = rare_mask.astype(int)

    # Let's verify and report positive rate
    pos_rate = df["task_rare"].mean()
    if pos_rate < 0.015 or pos_rate > 0.03:
        # Force exact tuning
        n_pos_target = int(n_total * 0.022)
        # Sort by collision risk score and pick top n_pos_target
        risk_score = -df["lead_dist"] - df["ego_accel"] - df["lead_rel_v"]
        threshold_val = np.sort(risk_score)[-n_pos_target]
        df["task_rare"] = (risk_score >= threshold_val).astype(int)

    print(f"Generated {n_logs} logs, total rows: {n_total}, rare positive rate: {df['task_rare'].mean():.4f}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-logs", type=int, default=1200)
    parser.add_argument("--out", type=str, default="data/raw.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_logs(n_logs=args.n_logs, seed=args.seed)
    df.to_parquet(args.out)
    print(f"Saved raw logs to {args.out}")
