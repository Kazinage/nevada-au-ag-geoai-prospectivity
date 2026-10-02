"""Decision-focused evaluation for prospectivity ranking."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def pooled_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def topk_capture(y, score, fractions=(0.01, 0.02, 0.05, 0.10)):
    """Fraction of observed positives captured by top-k fraction of supported area."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score)
    total = max(int(y.sum()), 1)
    out = {}
    for frac in fractions:
        n = max(1, int(np.ceil(len(y) * frac)))
        out[float(frac)] = float(y[order[:n]].sum() / total)
    return out


def bagging_dispersion(probabilities):
    """Mean prospectivity and ensemble standard deviation across B models."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2:
        raise ValueError("probabilities must have shape (n_models, n_samples)")
    return p.mean(axis=0), p.std(axis=0, ddof=1)
