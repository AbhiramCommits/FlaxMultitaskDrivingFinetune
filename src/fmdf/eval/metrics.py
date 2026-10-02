"""Rigorous evaluation metrics with bootstrap CIs for PR-AUC.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(labels: dict, preds: dict, n_bootstrap: int = 1000) -> dict:
    """Compute all evaluation metrics and bootstrap 95% CIs for rare PR-AUC.
    """
    # Event metrics
    event_true = labels["event"]
    event_pred = np.argmax(preds["event"], axis=-1)
    event_macro_f1 = float(f1_score(event_true, event_pred, average="macro"))
    per_class_f1 = f1_score(event_true, event_pred, average=None).tolist()

    # Horizon metrics
    horiz_true = labels["horizon"]
    horiz_pred = preds["horizon"]
    mae = float(mean_absolute_error(horiz_true, horiz_pred))
    rmse = float(np.sqrt(mean_squared_error(horiz_true, horiz_pred)))
    r2 = float(r2_score(horiz_true, horiz_pred))

    # Rare metrics
    rare_true = labels["rare"]
    rare_logits = preds["rare"]
    rare_probs = 1.0 / (1.0 + np.exp(-rare_logits))

    pr_auc = float(average_precision_score(rare_true, rare_probs))
    roc_auc = float(roc_auc_score(rare_true, rare_probs))

    # Precision@k and recall at fixed precision operating point
    sorted_idx = np.argsort(rare_probs)[::-1]
    k = max(1, int(0.02 * len(rare_true)))
    precision_at_k = float(np.mean(rare_true[sorted_idx[:k]]))

    # Bootstrap CIs for rare PR-AUC
    rng = np.random.default_rng(42)
    boot_pr_aucs = []
    n = len(rare_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if rare_true[idx].sum() > 0:
            b_pr = average_precision_score(rare_true[idx], rare_probs[idx])
            boot_pr_aucs.append(b_pr)

    if boot_pr_aucs:
        pr_auc_ci = [float(np.percentile(boot_pr_aucs, 2.5)), float(np.percentile(boot_pr_aucs, 97.5))]
    else:
        pr_auc_ci = [pr_auc, pr_auc]

    return {
        "event_macro_f1": event_macro_f1,
        "event_per_class_f1": per_class_f1,
        "horizon_mae": mae,
        "horizon_rmse": rmse,
        "horizon_r2": r2,
        "rare_pr_auc": pr_auc,
        "rare_pr_auc_ci": pr_auc_ci,
        "rare_roc_auc": roc_auc,
        "rare_precision_at_k": precision_at_k,
    }
