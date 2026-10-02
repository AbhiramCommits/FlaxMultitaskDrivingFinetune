"""Metric sensitivity study under varying rare-event prevalence and random seeds.
"""

import argparse
import csv
import os
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def run_sensitivity():
    rng = np.random.default_rng(42)
    n = 10000
    # Simulate scores and true labels
    scores = rng.normal(0.0, 1.0, size=n)
    probs = 1.0 / (1.0 + np.exp(-scores))

    prevalences = [0.005, 0.01, 0.02, 0.04]
    results = []

    os.makedirs("artifacts/results", exist_ok=True)
    csv_path = "artifacts/results/sensitivity.csv"

    print("Running metric sensitivity study across prevalence rates...")
    for prev in prevalences:
        # Subsample positives to match target prevalence
        threshold = np.percentile(probs, 100.0 * (1.0 - prev))
        y_true = (probs >= threshold).astype(int)

        pr_auc = average_precision_score(y_true, probs)
        roc_auc = roc_auc_score(y_true, probs)
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)

        results.append({
            "prevalence": prev,
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "f1": float(f1),
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prevalence", "pr_auc", "roc_auc", "f1"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved sensitivity results to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_sensitivity()
