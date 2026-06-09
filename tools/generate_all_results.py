#!/usr/bin/env python3
"""
generate_all_results.py

Regenerate all manuscript figures from source data, then organize
into results/main/ (main paper) and results/supplement/ (supplement).

Execution order follows the detect-localize-quantify narrative:
  1. Fig 3: DTC clustering + BOCPD changepoint posteriors  (detect)
  2. Fig 4: Clustering detail + changepoint detail          (localize)
  3. Fig 5bc: KGE member variants (best_kge -> Fig6 main,
              median_kge -> FigS2 supplement)               (quantify)
  4. Fig 5:  Ensemble median predictions (-> FigS1 supplement)
  5. Fig 5:  Transition-window robustness (-> main)         (quantify)
  6. Fig 7:  MODFLOW coupled groundwater                    (quantify)
  7. Copy:   Figures + summary tables -> main/ and supplement/

Figs 1-2 are manual diagrams (methodology + study area map) with no
generating script; they are copied directly from results/figures/.

Usage:
    python tools/generate_all_results.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
FIGURES = ROOT / "results" / "figures"
MAIN = ROOT / "results" / "main"
SUPP = ROOT / "results" / "supplement"

# Narrative order: detect -> localize -> quantify
PLOT_SCRIPTS = [
    "plot_fig3_clustering.py",
    "plot_fig3_changepoint.py",
    "plot_fig4_clustering.py",
    "plot_fig4_changepoint.py",
    "plot_fig5bc_kge_members.py",
    "plot_fig5_m1m2.py",
    "plot_fig5_transition_window.py",
    "plot_fig7_modflow.py",
]


def run_script(name: str) -> bool:
    """Run a plot script as a subprocess; return True if exit code == 0."""
    print(f"  Running {name}...", flush=True)
    result = subprocess.run(
        [sys.executable, str(TOOLS / name)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"  ERROR in {name}:\n{result.stderr}", flush=True)
        return False
    print(f"  OK: {name}", flush=True)
    return True


def reset_dir(path: Path) -> None:
    """Remove and recreate a directory for idempotent reruns."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_step() -> None:
    """Copy figures and CSVs into results/main/ and results/supplement/."""
    reset_dir(MAIN)
    reset_dir(SUPP)

    main_fig_count = 0
    main_csv_count = 0
    supp_fig_count = 0
    supp_csv_count = 0

    print("\nCopying to results/main/...", flush=True)

    # --- Main paper figures ---

    # Fig 1 + Fig 2: manual diagrams (PDF only, no generating script)
    for name in ("Fig1_Methodology.pdf", "Fig2_US_RRCA.pdf"):
        shutil.copy2(FIGURES / name, MAIN / name)
        print(f"  {name}", flush=True)
        main_fig_count += 1

    # Fig 3-5, 7: direct copy (PDF + PNG)
    direct_figures = [
        "Fig3_clustering",
        "Fig4_changepoint",
        "Fig5_transition_window",
        "Fig7_modflow_coupled",
    ]
    for stem in direct_figures:
        for ext in ("pdf", "png"):
            src = FIGURES / f"{stem}.{ext}"
            dst = MAIN / f"{stem}.{ext}"
            shutil.copy2(src, dst)
            print(f"  {stem}.{ext}", flush=True)
            main_fig_count += 1

    # Fig 6: renamed from Fig5c_m1m2_best_kge (per D-06)
    for ext in ("pdf", "png"):
        src = FIGURES / f"Fig5c_m1m2_best_kge.{ext}"
        dst = MAIN / f"Fig6_irrigation.{ext}"
        shutil.copy2(src, dst)
        print(f"  Fig6_irrigation.{ext}  (renamed from Fig5c_m1m2_best_kge.{ext})", flush=True)
        main_fig_count += 1

    # --- Main paper CSVs ---
    csv_main_mappings = [
        (ROOT / "results" / "transition_window" / "summary.csv",
         MAIN / "summary.csv"),
        (ROOT / "results" / "modflow_propagation" / "uncertainty_summary.csv",
         MAIN / "uncertainty_summary.csv"),
        (ROOT / "results" / "modflow_propagation" / "coupled_ensemble" / "coupled_waterhead_summary.csv",
         MAIN / "coupled_waterhead_summary.csv"),
    ]
    for src, dst in csv_main_mappings:
        shutil.copy2(src, dst)
        print(f"  {dst.name}", flush=True)
        main_csv_count += 1

    print(f"\n  -> {main_fig_count} figure files + {main_csv_count} CSVs to main/", flush=True)

    # --- Supplement figures ---
    print("\nCopying to results/supplement/...", flush=True)

    # FigS1: renamed from Fig5_m1m2_predictions (per D-05)
    for ext in ("pdf", "png"):
        src = FIGURES / f"Fig5_m1m2_predictions.{ext}"
        dst = SUPP / f"FigS1_m1m2_predictions.{ext}"
        shutil.copy2(src, dst)
        print(f"  FigS1_m1m2_predictions.{ext}  (renamed from Fig5_m1m2_predictions.{ext})", flush=True)
        supp_fig_count += 1

    # FigS2: renamed from Fig5b_m1m2_ens_kge_median (per D-05)
    for ext in ("pdf", "png"):
        src = FIGURES / f"Fig5b_m1m2_ens_kge_median.{ext}"
        dst = SUPP / f"FigS2_m1m2_ens_kge_median.{ext}"
        shutil.copy2(src, dst)
        print(f"  FigS2_m1m2_ens_kge_median.{ext}  (renamed from Fig5b_m1m2_ens_kge_median.{ext})", flush=True)
        supp_fig_count += 1

    # --- Supplement CSVs ---
    csv_supp_mappings = [
        (ROOT / "results" / "transition_window" / "per_agent_detail.csv",
         SUPP / "per_agent_detail.csv"),
        (ROOT / "results" / "transition_window" / "pi_decomposition.csv",
         SUPP / "pi_decomposition.csv"),
        (ROOT / "results" / "transition_window" / "leakage_year_ablation.csv",
         SUPP / "leakage_year_ablation.csv"),
        (ROOT / "results" / "modflow_propagation" / "rrca_export" / "rrca_summary.csv",
         SUPP / "rrca_summary.csv"),
    ]
    for src, dst in csv_supp_mappings:
        shutil.copy2(src, dst)
        print(f"  {dst.name}", flush=True)
        supp_csv_count += 1

    print(f"\n  -> {supp_fig_count} figure files + {supp_csv_count} CSVs to supplement/", flush=True)

    total_figs = main_fig_count + supp_fig_count
    total_csvs = main_csv_count + supp_csv_count
    print(f"\nDone. {total_figs} figure files, {total_csvs} CSVs organized.", flush=True)


def main() -> None:
    """Run all plot scripts, then organize outputs into main/ and supplement/."""
    print("Generating figures (8 scripts)...\n", flush=True)

    errors = []
    for script in PLOT_SCRIPTS:
        if not run_script(script):
            errors.append(script)

    if errors:
        print(f"\nWARNING: {len(errors)} script(s) failed: {errors}", flush=True)
        print("Continuing with copy step (using existing figures)...", flush=True)

    print("\nOrganizing outputs...", flush=True)
    copy_step()


if __name__ == "__main__":
    main()
