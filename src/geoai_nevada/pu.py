"""Positive-unlabeled helpers following the Elkan-Noto correction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


@dataclass
class PUModel:
    estimator: object
    selection_constant: float

    def predict_proba(self, X):
        p_selected = self.estimator.predict_proba(X)[:, 1]
        return np.clip(p_selected / self.selection_constant, 0.0, 1.0)


def fit_elkan_noto(
    estimator,
    X,
    s,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> PUModel:
    """Fit P-vs-U classifier and estimate c=P(S=1|Y=1) from inner OOF scores."""
    X = np.asarray(X)
    s = np.asarray(s, dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = cross_val_predict(
        clone(estimator), X, s, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    positive_scores = oof[s == 1]
    if len(positive_scores) == 0:
        raise ValueError("At least one labelled positive is required")
    c = float(np.clip(positive_scores.mean(), 1e-6, 1.0))
    fitted = clone(estimator).fit(X, s)
    return PUModel(fitted, c)
