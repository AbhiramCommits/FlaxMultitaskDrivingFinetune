"""Calibration metrics (ECE, MCE, Brier score) and temperature scaling.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def compute_calibration(y_true: np.y_array if hasattr(np, 'y_array') else np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> dict:
    """Compute ECE (equal-mass bins), MCE, and Brier score.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    brier = float(np.mean((y_prob - y_true) ** 2))

    # Equal-mass bins (quantiles)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_prob, quantiles)
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0

    ece = 0.0
    mce = 0.0
    n = len(y_true)

    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask)
            cal_diff = abs(acc - conf)
            ece += (bin_size / n) * cal_diff
            mce = max(mce, cal_diff)
            bin_accs.append(float(acc))
            bin_confs.append(float(conf))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
    }


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit optimal temperature scalar on validation set logits.
    """
    best_T = 1.0
    best_ece = float("inf")
    for T in np.linspace(0.5, 3.0, 51):
        scaled_logits = logits / T
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        cal = compute_calibration(labels, probs)
        if cal["ece"] < best_ece:
            best_ece = cal["ece"]
            best_T = float(T)
    return best_T


def plot_reliability_curve(y_true_before, y_prob_before, y_true_after, y_prob_after, out_path="artifacts/figs/reliability.png"):
    cal_before = compute_calibration(y_true_before, y_prob_before)
    cal_after = compute_calibration(y_true_after, y_prob_after)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(cal_before["bin_confs"], cal_before["bin_accs"], "marker='o'", label=f"Before (ECE={cal_before['ece']:.3f})")
    plt.plot(cal_after["bin_confs"], cal_after["bin_accs"], "marker='s'", label=f"After (ECE={cal_after['ece']:.3f})")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram (Temperature Scaling)")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
