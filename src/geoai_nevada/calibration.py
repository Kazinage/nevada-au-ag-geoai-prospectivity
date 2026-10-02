"""Leakage-safe post-hoc probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


@dataclass
class ScoreCalibrator:
    method: str
    model: object

    def predict(self, scores):
        x = np.asarray(scores, dtype=float).reshape(-1)
        if self.method == "isotonic":
            return np.clip(self.model.predict(x), 0.0, 1.0)
        return self.model.predict_proba(x.reshape(-1, 1))[:, 1]


def _fit(method, scores, y):
    if method == "isotonic":
        return ScoreCalibrator(
            method,
            IsotonicRegression(out_of_bounds="clip").fit(scores, y),
        )
    if method == "platt":
        return ScoreCalibrator(
            method,
            LogisticRegression(max_iter=2000).fit(np.asarray(scores).reshape(-1, 1), y),
        )
    raise ValueError(method)


def select_calibrator(scores, y, *, n_splits=5, random_state=42):
    """Choose Platt or isotonic mapping by cross-validated Brier score."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    losses = {"platt": [], "isotonic": []}
    for tr, va in cv.split(scores, y):
        for method in losses:
            cal = _fit(method, scores[tr], y[tr])
            losses[method].append(brier_score_loss(y[va], cal.predict(scores[va])))
    best = min(losses, key=lambda k: np.mean(losses[k]))
    return _fit(best, scores, y), {k: float(np.mean(v)) for k, v in losses.items()}
