"""Spatial blocking and train-test exclusion helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def block_ids(xy, block_size_m: float):
    """Assign deterministic square spatial blocks from projected coordinates."""
    xy = np.asarray(xy, dtype=float)
    origin = np.nanmin(xy, axis=0)
    ij = np.floor((xy - origin) / float(block_size_m)).astype(int)
    return np.array([f"{i}_{j}" for i, j in ij], dtype=object)


def exclusion_mask(train_xy, test_xy, radius_m: float):
    """Return mask of training rows at least radius_m from every test point."""
    train_xy = np.asarray(train_xy, dtype=float)
    test_xy = np.asarray(test_xy, dtype=float)
    if len(test_xy) == 0:
        return np.ones(len(train_xy), dtype=bool)
    tree = cKDTree(test_xy)
    distance, _ = tree.query(train_xy, k=1, workers=-1)
    return distance >= float(radius_m)


def thin_points(xy, radius_m: float):
    """Greedy spatial thinning for occurrence inventories."""
    xy = np.asarray(xy, dtype=float)
    keep = []
    accepted = []
    for i, point in enumerate(xy):
        if not accepted:
            keep.append(i)
            accepted.append(point)
            continue
        d = np.sqrt(((np.asarray(accepted) - point) ** 2).sum(axis=1))
        if np.all(d >= radius_m):
            keep.append(i)
            accepted.append(point)
    return np.asarray(keep, dtype=int)
