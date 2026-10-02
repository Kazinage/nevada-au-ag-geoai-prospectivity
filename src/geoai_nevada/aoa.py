"""Area-of-Applicability utilities for fold-wise geoscience ML."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


@dataclass
class AOAResult:
    dissimilarity: np.ndarray
    inside: np.ndarray
    threshold: float


class AreaOfApplicability:
    """Nearest-neighbour AOA in robustly scaled feature space.

    The training cloud is robustly standardized (median/IQR), a shrinkage
    covariance is estimated, and Mahalanobis nearest-neighbour distance is
    used as the dissimilarity index. The AOA threshold is the requested
    quantile of leave-one-out training distances.
    """

    def __init__(self, quantile: float = 0.95):
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = quantile

    def fit(self, X: np.ndarray) -> "AreaOfApplicability":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("X must be a 2-D array with at least 3 rows")
        self.scaler_ = RobustScaler(quantile_range=(25, 75)).fit(X)
        Z = self.scaler_.transform(X)
        self.precision_ = LedoitWolf().fit(Z).precision_
        self.nn_ = NearestNeighbors(
            n_neighbors=2,
            metric="mahalanobis",
            metric_params={"VI": self.precision_},
            algorithm="brute",
        ).fit(Z)
        d, _ = self.nn_.kneighbors(Z, n_neighbors=2)
        loo = d[:, 1]
        self.threshold_ = float(np.quantile(loo, self.quantile))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        d, _ = self.nn_.kneighbors(Z, n_neighbors=1)
        return d[:, 0]

    def predict(self, X: np.ndarray) -> AOAResult:
        di = self.score(X)
        return AOAResult(di, di <= self.threshold_, self.threshold_)
