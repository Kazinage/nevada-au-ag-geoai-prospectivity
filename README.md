# Nevada Au-Ag GeoAI Prospectivity

**Spatially honest, uncertainty-aware mineral prospectivity mapping for Au-Ag in the Nevada Great Basin.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fmin16090886-blue)](https://doi.org/10.3390/min16090886)

This repository is a cleaned, portfolio-oriented implementation of the workflow developed for:

> **Saduov, A.; Zholtayev, G.; Togizov, K.; Umarbekova, Z.; Zhumabay, N.; Amangeldi, N. (2026).**  
> *AOA-Constrained, Calibrated Random-Forest Prospectivity Mapping for Au-Ag in Nevada Great Basin: Spatially Independent Validation, Uncertainty, and Decision-Focused Top-k Targets.*  
> **Minerals, 16(9), 886.** https://doi.org/10.3390/min16090886

## Why this project matters

Regional mineral prospectivity models can look convincing while still failing in deployment because of spatial leakage, unlabeled background data, extrapolation outside the training domain, and poorly calibrated probability scores. This project treats those issues as part of one modelling system rather than as separate post-processing steps.

The workflow combines:

- positive-unlabeled (PU) learning rather than assuming that unlabelled cells are barren;
- geographically separated five-fold validation with train-test exclusion buffers;
- fold-wise **Area of Applicability (AOA)** control in predictor space;
- Random Forest modelling with out-of-fold inference;
- post-hoc probability calibration selected by cross-validated Brier score;
- ensemble dispersion as an uncertainty layer;
- SHAP / partial-dependence diagnostics for geological interpretation;
- top-k area targeting for decision-focused exploration.

## Published results

The peer-reviewed study reported:

| Target | AOA coverage | OOF AP | OOF ROC-AUC | Calibrated Brier |
|---|---:|---:|---:|---:|
| Ag | 54.331% | 0.745 | 0.908 | 0.066 |
| Au | 73.118% | 0.777 | 0.912 | 0.059 |

The highest-ranked **10% of the supported AOA** captured **85.6% of Ag** and **84.2% of Au** pooled out-of-fold positives, corresponding to 8.6x and 8.4x targeting efficiencies relative to random spatial selection.

These values are reported results from the publication. The synthetic demo in this repository is only a software demonstration and is not intended to reproduce the paper's numerical results.

## Study design

```text
Public geoscience data
        |
        v
250 m harmonized predictor grid (EPSG:32611)
        |
        +--> potential fields: gravity / pseudogravity / HGM / match-filtered products
        +--> structural controls: Quaternary faults / TD / TS / TDTS
        +--> thermal control: conductive heat flow
        |
        v
Occurrence curation + spatial thinning
        |
        v
Spatial GroupKFold + train-test exclusion
        |
        v
Fold-wise Area of Applicability
        |
        v
Positive-Unlabeled Random Forest
        |
        v
Pooled out-of-fold predictions
        |
        +--> Platt vs isotonic calibration (CV Brier selection)
        +--> bagging dispersion / uncertainty
        +--> SHAP and PDP diagnostics
        |
        v
AOA-masked prospectivity + uncertainty + top-k targets
```

## Repository structure

```text
.
├── data/
│   └── README.md
├── docs/
│   └── methodology.md
├── examples/
│   └── synthetic_demo.py
├── src/
│   └── geoai_nevada/
│       ├── __init__.py
│       ├── aoa.py
│       ├── calibration.py
│       ├── evaluation.py
│       ├── pu.py
│       └── spatial.py
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/Kazinage/nevada-au-ag-geoai-prospectivity.git
cd nevada-au-ag-geoai-prospectivity
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -e .
python examples/synthetic_demo.py
```

## Data and reproducibility

The original study used public USGS/state-partner sources for heat flow, potential-field geophysics, slip/dilation tendency, Quaternary faults, regional geology, and mineral occurrences. The main sources include:

- USGS Nevada Geothermal ML heat-flow release: https://doi.org/10.5066/P9HRX1LR
- USGS Nevada Geothermal ML geophysics release: https://doi.org/10.5066/P9676O1M
- USGS slip/dilation tendency release: https://doi.org/10.5066/P9RM9A9B
- State Geologic Map Compilation: https://doi.org/10.3133/ds1052
- USGS MRDS / USMIN mineral occurrence resources.

Large source rasters, processed occurrence inventories, and study-specific intermediate products are **not redistributed here**. See [data/README.md](data/README.md) for the expected data contract.

## Scientific scope

This is a regional reconnaissance and exploration-targeting workflow. It does **not** estimate resources, reserves, grades, or deposit-scale economic viability. Calibrated scores are interpreted relative to the observed-positive inventory inside the supported AOA, not as absolute probabilities that an economic deposit exists in a grid cell.

## Status

The repository is a curated research implementation derived from the original analysis notebooks and the published methodology. It is intentionally cleaner and more modular than the exploratory code used during model development.

## Author

**Alisher Saduov, PhD**  
Geophysics, mineral exploration, GeoAI, spatial machine learning and uncertainty quantification  
Satbayev University, Kazakhstan  
ORCID: 0000-0003-1501-7772

## License

Code in this repository is released under the MIT License. External datasets and publication content remain subject to their respective licenses and terms of use.


## Related GeoAI projects

This repository is part of a focused geoscience/GeoAI portfolio:

- [Nevada Au-Ag GeoAI Prospectivity](https://github.com/Kazinage/nevada-au-ag-geoai-prospectivity) - spatial CV, PU learning, AOA, calibration and uncertainty.
- [Zambia Cu-Co Unsupervised Prospectivity](https://github.com/Kazinage/zambia-cu-co-unsupervised-prospectivity) - ensemble clustering for label-scarce airborne geophysics.
- [Uranium Horizon ML Geophysics](https://github.com/Kazinage/uranium-horizon-ml-geophysics) - grouped well-log validation and productive-horizon classification.
- [Mineral Anomaly Detection ML](https://github.com/Kazinage/mineral-anomaly-detection-ml) - PCA, Isolation Forest, One-Class SVM and anomaly clustering.
