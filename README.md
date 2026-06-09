# Non-stationary groundwater pumping behavior → coupled groundwater forecast uncertainty

Code to reproduce the analyses and figures in:

> Hu, Y., & Xi, S. *From pumping regime shifts to groundwater forecast uncertainty in coupled human–groundwater models.*

Annual groundwater pumping-depth records for 43 county-level "agents" in the High Plains Aquifer Hydrologic Observatory Area (Ogallala Aquifer), 1993–2020, are used to (1) detect *where* pumping behavior departs from stationarity, (2) localize *when* via Bayesian changepoint detection, (3) compare stationary (M1) vs. regime-aware (M2) data-driven pumping models, and (4) propagate the resulting behavioral uncertainty through the Republican River Compact Administration (RRCA) MODFLOW-2000 groundwater model.

## Method pipeline

`Deep Temporal Clustering (DTC)` → `Bayesian Online Changepoint Detection (BOCPD)` → `XGBoost M1 (stationary) vs. M2 (regime-aware)` → `100-member moving-block bootstrap ensembles` → `RRCA MODFLOW propagation`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.11 recommended
pip install -r requirements.txt
```

Pinned dependencies (all permissive-licensed): numpy, pandas, scipy, matplotlib, scikit-learn, xgboost, optuna, tensorflow, statsmodels, tslearn, bayesian-changepoint-detection, pytest.

## Data

Code and data are deposited separately (FAIR policy):

- **Input data** — HydroShare, CC-BY 4.0: *Groundwater Irrigation Data in Part of the Ogallala Aquifer* (Hu, 2024b). Download and place under `data/` (see `DATA_DICTIONARY.md`).
- **Third-party sources** (cite, not redistributed): precipitation & temperature from **GHCNd** (NOAA NCEI); crop prices from **USDA**; diesel price from **U.S. EIA**; target pumping depth from the **RRCA MODFLOW-2000** model (McKusick, 2003).

Expected layout after downloading the data:

```
data/
  agentdata_{1..48}.csv                 # 43 valid agents; monthly panel
  irrigation_depth_monthly_1993_2020.csv
  irrigation_depth_annual_1993_2020.csv
  annual_irrigation_depth.csv
  monthly_crop_data9320.csv             # USDA prices + EIA diesel
  prcp4rrca9320/                        # gridded precip (from GHCNd)
  temp4rrca9320/                        # gridded temperature (from GHCNd)
  agRatio/                              # RRCA-format MODFLOW inputs (+ M1/M2 variants)
```

## Reproducing the figures

`generate_all_results.py` runs the **plotting** step only; the plot scripts read intermediate CSVs produced by the **analysis** scripts. Run the full chain in order:

```bash
# 1. Build annual series from the monthly agent panel
python tools/aggregate_annual_irrigation.py

# 2. Detect — temporal clustering of pumping trajectories
python tools/run_dtc_clustering.py
python tools/evaluate_dtc_robustness.py        # consensus / silhouette diagnostics
python tools/run_clustering_baselines.py       # k-means / hierarchical baselines

# 3. Localize — Bayesian changepoint detection
python tools/run_changepoint_detection.py
python tools/run_changepoint_robustness.py     # 27-config sensitivity sweep

# 4. Quantify — stationary (M1) vs regime-aware (M2) pumping models
python tools/run_xgboost_abm.py                # pooled Cluster-2 model
python tools/run_xgboost_c1.py                 # Cluster-1 regime models
python tools/run_transition_window.py          # CP-1 / CP / CP+1 robustness

# 5. Propagate — behavioral uncertainty through RRCA MODFLOW
python tools/run_modflow_ensemble.py
python tools/export_rrca.py

# 6. Generate all manuscript figures (Fig 3-7 + SI)
python tools/generate_all_results.py
```

### Figure → script map

| Figure | Content | Script |
|--------|---------|--------|
| Fig 1 | Methodology diagram | manual (no script) |
| Fig 2 | Study-area map (HPA-HOA / RRCA) | manual (no script) |
| Fig 3 | Clustering + cluster-level changepoint diagnostics | `plot_fig3_clustering.py` |
| Fig 4 | Agent-level changepoint localization (8 agents) | `plot_fig4_changepoint.py` |
| Fig 5 | Transition-window robustness (CP-1 / CP / CP+1) | `plot_fig5_transition_window.py` |
| Fig 6 | M1 vs M2 best-KGE pumping-depth members | `plot_fig5bc_kge_members.py` (best_kge) |
| Fig 7 | Coupled MODFLOW groundwater-level change | `plot_fig7_modflow.py` |
| Fig S1 | M1 vs M2 ensemble-median predictions | `plot_fig5_m1m2.py` |
| Fig S2 | M1 vs M2 median-KGE members | `plot_fig5bc_kge_members.py` (median_kge) |

## Reproducibility settings

- `MASTER_SEED = 42` (root seed for Optuna, bootstrap, framework RNGs)
- Bootstrap ensemble: `N_BOOT = 100` (200 for the Cluster-2 ABM sensitivity), constrained moving-block bootstrap, `BLOCK_SIZE = 3`
- Hyperparameter tuning: Optuna, 100 TPE trials
- Prediction intervals: 90% = [5th, 95th] percentile of the ensemble
- Non-stationary agents analyzed: **2, 12, 14, 18, 20, 24, 28, 29**

## Third-party code

`tools/dtc/` is the **Deep Temporal Clustering** implementation by **Florent Forest** (`github.com/FlorentF9/DeepTemporalClustering`), redistributed under its upstream **MIT License** (see `tools/dtc/NOTICE`). All other code is original to this project.

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. Please cite both the software (Zenodo DOI) and the dataset (HydroShare, Hu 2024b).
