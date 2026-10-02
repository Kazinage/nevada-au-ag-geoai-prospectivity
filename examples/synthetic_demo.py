"""Executable synthetic demonstration.

This is not the Nevada dataset. It demonstrates the software architecture
without redistributing study-specific processed data.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from geoai_nevada.aoa import AreaOfApplicability
from geoai_nevada.calibration import select_calibrator
from geoai_nevada.evaluation import pooled_metrics, topk_capture
from geoai_nevada.pu import fit_elkan_noto
from geoai_nevada.spatial import block_ids, exclusion_mask


rng = np.random.default_rng(42)
n = 5000
xy = rng.uniform(0, 100_000, size=(n, 2))
X = np.column_stack([
    np.sin(xy[:, 0] / 14000) + rng.normal(0, 0.25, n),
    np.cos(xy[:, 1] / 17000) + rng.normal(0, 0.25, n),
    rng.normal(size=n),
    rng.normal(size=n),
])
latent = 1.4 * X[:, 0] - 0.8 * X[:, 1] + 0.35 * X[:, 2]
y_true = (latent > np.quantile(latent, 0.90)).astype(int)

s = np.zeros(n, dtype=int)
positive_idx = np.flatnonzero(y_true)
s[rng.choice(positive_idx, size=int(0.55 * len(positive_idx)), replace=False)] = 1

groups = block_ids(xy, block_size_m=20_000)
gkf = GroupKFold(n_splits=5)
oof = np.full(n, np.nan)

for fold, (tr, va) in enumerate(gkf.split(X, s, groups), start=1):
    safe = exclusion_mask(xy[tr], xy[va], radius_m=5_000)
    tr = tr[safe]

    aoa = AreaOfApplicability(quantile=0.95).fit(X[tr])
    supported = aoa.predict(X[va]).inside

    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", random_state=fold, n_jobs=-1
    )
    pu = fit_elkan_noto(rf, X[tr], s[tr], random_state=fold)
    oof[va[supported]] = pu.predict_proba(X[va][supported])

mask = np.isfinite(oof)
cal, cv_brier = select_calibrator(oof[mask], s[mask])
p = cal.predict(oof[mask])

print("Calibrator CV Brier:", cv_brier)
print("OOF metrics against observed positives:", pooled_metrics(s[mask], p))
print("Top-k capture:", topk_capture(s[mask], p))
