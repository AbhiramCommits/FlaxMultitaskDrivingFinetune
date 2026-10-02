"""DuckDB SQL dataset assembly and splitting.
"""

import argparse
import os
import duckdb
import pandas as pd


def build_dataset(raw_parquet: str = "data/raw.parquet", out_dir: str = "data"):
    con = duckdb.connect(database=":memory:")
    
    # Register raw table
    con.execute(f"CREATE TABLE raw AS SELECT * FROM read_parquet('{raw_parquet}')")

    # Run SQL queries
    with open("queries/assemble_windows.sql", "r") as f:
        assemble_sql = f.read()
    con.execute(assemble_sql)

    with open("queries/split_by_log.sql", "r") as f:
        split_sql = f.read()
    con.execute(split_sql)

    os.makedirs(out_dir, exist_ok=True)
    
    # Export splits
    for split_name in ["train", "val", "test"]:
        df_split = con.execute(f"SELECT * FROM windows_split WHERE split = '{split_name}'").fetchdf()
        # Filter windows that have exactly length 32
        # Let's inspect sequence lengths
        out_path = os.path.join(out_dir, f"windows_{split_name}.parquet")
        df_split.to_parquet(out_path)
        print(f"Saved {len(df_split)} windows to {out_path}")

    # Print class stats
    with open("queries/class_stats.sql", "r") as f:
        stats_sql = f.read()
    stats_df = con.execute(stats_sql).fetchdf()
    print("\n=== Class Stats ===")
    print(stats_df.to_string())

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw.parquet")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.build:
        build_dataset(args.raw, args.out)
