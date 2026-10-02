# Data contract

Large geospatial source files and processed study products are intentionally not committed to this repository.

## Public source families used in the study

The published workflow integrated public geoscience sources including:

1. conductive heat flow from the Nevada Geothermal Machine Learning project;
2. gravity, pseudogravity, match-filtered and horizontal-gradient products;
3. slip tendency, dilation tendency and TDTS structural information;
4. Quaternary fault traces;
5. regional geology;
6. public Au and Ag occurrence inventories.

See the main README and the peer-reviewed article for source citations and dataset DOIs.

## Expected tabular model input

Reusable modelling utilities expect a table with:

- projected coordinates `X`, `Y` in metres;
- one or more continuous predictor columns;
- an observed-positive indicator `s` (1 = labelled occurrence, 0 = unlabeled);
- a spatial group/block identifier for outer validation.

A minimal example:

```text
X,Y,gravity_hgm,pseudogravity_hgm,heat_flow,dist_qfault_km,td_near,ts_near,s,group
...
```

## Reproducibility note

The paper's exact numerical results depend on the curated occurrence inventory, fold definitions, harmonized predictor stack, AOA masks and tuned model settings. The code here provides the cleaned computational building blocks and an executable synthetic example. It does not redistribute study-specific processed data.
