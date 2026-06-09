# Code deposit manifest

What goes into the public code repository (Zenodo/GitHub) and what stays out.
`assemble_release.sh` builds `release/gw-pumping-regime-shifts/tools/` from the working `tools/` using exactly these rules.

## INCLUDE — analysis & figure pipeline (`tools/`)

**Orchestration**
- `generate_all_results.py` — runs the plot scripts, organizes `results/main` + `results/supplement`

**Data preparation**
- `aggregate_annual_irrigation.py`, `update_irrigation_depth.py`

**Detect — temporal clustering**
- `run_dtc_clustering.py`, `evaluate_dtc_robustness.py`, `cluster_robustness.py`,
  `run_clustering_baselines.py`, `cluster_irrigation_summary.py`
- `dtc/` — vendored DTC (`__init__.py`, `DeepTemporalClustering.py`, `TAE.py`, `TSClusteringLayer.py`, `tsdistances.py`) + generated `NOTICE`

**Localize — changepoint detection**
- `run_changepoint_detection.py`, `run_changepoint_robustness.py`

**Quantify — M1 vs M2 prediction**
- `run_xgboost_abm.py`, `run_xgboost_c1.py`, `run_xgboost_agent12.py`,
  `run_xgboost_agent20.py`, `run_xgboost_agent12_precp.py`, `run_transition_window.py`

**Propagate — MODFLOW ensemble**
- `run_modflow_ensemble.py`, `export_rrca.py`, `run_pi_decomposition.py`

**Validation / leakage**
- `validate_leakage_pipeline.py`, `validate_leakage_transition_window.py`, `check_year_leakage.py`

**Plotting**
- `plot_fig3_clustering.py`, `plot_fig3_changepoint.py`, `plot_fig4_clustering.py`,
  `plot_fig4_changepoint.py`, `plot_fig5_m1m2.py`, `plot_fig5bc_kge_members.py`,
  `plot_fig5_transition_window.py`, `plot_fig7_modflow.py`
- Auxiliary: `plot_annual_irrigation.py`, `plot_cp_point_estimates.py`

**Tests**
- `tools/tests/` (`conftest.py`, `test_modflow_ensemble.py`) → `tools/tests/`
- repo-root `tests/` (`test_constrained_bootstrap.py`, `test_two_regime.py`) → deposit-root `tests/`
  (run with `PYTHONPATH=tools`)

## EXCLUDE — not part of the published analysis

- `convert_md_to_docx.py`, `convert_paper_to_docx.py`, `generate_review_response.py`
  — manuscript-production utilities, irrelevant to reproducing results.
- `data/`, `results/`, `outputs/`, figures — deposited separately / regenerated.
- `.env`, secrets, `__pycache__/`, virtualenvs, `manuscript/`, `.planning/`.

## NOTE — keep the folder named `tools/`

The scripts resolve paths as `ROOT = Path(__file__).resolve().parent.parent` and read
`ROOT/tools`, `ROOT/data`, `ROOT/results`. Renaming `tools/` would break cross-script
calls in `generate_all_results.py`. The deposit therefore mirrors the working layout:
`README.md`, `LICENSE`, `requirements.txt`, `CITATION.cff`, and `tools/` at the repo root;
`data/` and `results/` are created by the user/run.
