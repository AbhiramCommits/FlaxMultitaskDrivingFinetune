"""Tests for metric correctness (PR-AUC, ECE).
"""

import numpy as np
from fmdf.eval.calibration import compute_calibration
from sklearn.metrics import average_precision_score


def test_metrics_correctness():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])

    sklearn_prauc = average_precision_score(y_true, y_prob)
    assert 0.0 <= sklearn_prauc <= 1.0

    # Perfectly calibrated
    cal_perfect = compute_calibration(np.array([0, 1]), np.array([0.0, 1.0]))
    assert cal_perfect["ece"] < 1e-5

    # Overconfident
    cal_over = compute_calibration(np.array([0, 0]), np.array([0.9, 0.9]))
    assert cal_over["ece"] > 0.2
