# Methodology notes

This repository follows the logic of the published Nevada Au-Ag study while separating reusable software components from study-specific data preparation.

## 1. Spatial support and labels

All metric spatial operations in the study used WGS 84 / UTM Zone 11N (EPSG:32611). Public rasters were harmonized to a 250 m analysis grid. Au and Ag were modelled separately.

Occurrence inventories are positive observations. Cells without documented occurrences are **unlabeled**, not verified barren cells. The published workflow therefore uses a positive-unlabeled formulation.

Occurrence thinning and spatial fold design were informed by nearest-neighbour distances. The final study used five geographically separated folds, with a 10 km train-test exclusion distance. Commodity-specific spatial block radii were 14 km for Ag and 20 km for Au.

## 2. Area of Applicability

AOA is estimated inside every outer training fold. Predictors are robustly standardized using the training fold only. A feature-space dissimilarity index is calculated from prediction cells to the training cloud. The operational threshold is the 0.95 quantile of leave-one-out training dissimilarities.

The key principle is **fold-wise fitting**: the held-out fold cannot influence scaling, covariance estimation, the AOA threshold, model fitting, or calibration.

## 3. Positive-unlabeled learning

For each outer fold, a probabilistic classifier estimates the probability that a sample is selected/labeled. The selection constant is estimated from internal out-of-fold predictions on labelled positives, following the Elkan-Noto logic. Corrected scores are clipped to [0, 1].

The implementation in this repository exposes this step as a reusable component rather than hard-coding Nevada-specific feature names.

## 4. Base learner

The publication used Random Forest as the principal base learner because it is robust to nonlinear interactions and partially collinear predictors, works naturally with bagging-based uncertainty, and remains compatible with calibration and SHAP interpretation.

The reported reference configuration used 500 trees with `max_features="sqrt"` unless otherwise noted in the study's tuning procedure.

## 5. Calibration

Calibration is performed **after** base-model selection on pooled out-of-fold scores. Platt scaling and isotonic regression are compared using internal cross-validated Brier score. The better mapping is refitted on all pooled OOF predictions for the commodity.

In the publication, isotonic regression was selected for both Ag and Au.

## 6. Uncertainty

Predictive dispersion is estimated from repeated RF bagging members. For every supported cell, the standard deviation of calibrated probabilities is reported alongside mean prospectivity. This is a model-dispersion measure, not a complete decomposition of geological uncertainty.

## 7. Decision-focused evaluation

The primary operational products are AOA-masked top-k area slices. Ranking quality is evaluated by the fraction of observed positives captured in the highest-ranked 1%, 2%, 5%, and 10% of supported area.

The published 10% AOA slice captured 85.6% of Ag and 84.2% of Au pooled-OOF positives.

## 8. What this repository does not claim

- The synthetic example does not reproduce the paper's results.
- Unlabeled cells are not interpreted as confirmed negatives.
- Calibrated scores are not deposit-resource probabilities.
- The workflow is regional reconnaissance, not reserve estimation.
- Transferring the model to another province requires re-estimating spatial support, AOA, labels, and calibration.
