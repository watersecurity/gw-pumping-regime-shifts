# Data dictionary — HydroShare deposit (Hu, 2026b)

Input data to reproduce Hu & Xi. Place these under `data/` in the code
repository. License: **CC-BY 4.0**. Coverage: **1993–2020**, High Plains Aquifer Hydrologic
Observatory Area (Ogallala). Irrigation season months **May–October**.

## Provenance key
- **P** = project-generated / derived (share here)
- **D(src)** = derived from third-party source `src` (share derived form; cite original)
- **3P(src)** = essentially third-party `src` (consider citing instead of re-hosting)

## Files

| File / folder | Contents | Units | Provenance |
|---|---|---|---|
| `agentdata_{1..48}.csv` (43 files) | Monthly per-agent panel: irrigation depth, crop acreage (corn, wheat, soybeans, sorghum), diesel price, precipitation, temperature, year, month | mm; acres; US$/gal; mm; °C | **P** (irrigation) + **D(GHCNd, USDA, USEIA)** (covariates) |
| `irrigation_depth_monthly_1993_2020.csv` | Monthly irrigation depth by agent/year/month | mm, ft | **P** (from RRCA pumping) |
| `irrigation_depth_annual_1993_2020.csv` | Annual pumping volume, irrigated area, depth | acre-ft; acres; ft, mm, in | **P** |
| `annual_irrigation_depth.csv` | Wide annual depth, agent × year (1993–2020) | mm | **P** (`aggregate_annual_irrigation.py`) |
| `monthly_crop_data9320.csv` (+ `.xlsx`) | Monthly commodity prices (corn/soybean/sorghum/wheat) + diesel | US$/bushel; US$/gal | **D(USDA, USEIA)** |
| `prcp4rrca9320/monthlyP_*.csv` | Monthly precipitation per RRCA gridcell | mm | **D(GHCNd)** |
| `temp4rrca9320/monthlyT_*.csv` | Monthly temperature per RRCA gridcell (Avg/Min/Max) | °C | **D(GHCNd)** |
| `agRatio/agAreaR.YYYY` (×28) | Agent → agricultural-area ratio per RRCA gridcell, annual | dimensionless | **3P(RRCA)** — not in deposit; cite RRCA |
| `agRatio/agWatR.YYYY` (×28) | Agent → water amount per gridcell, annual | acre-ft | **3P(RRCA)** — not in deposit; cite RRCA |
| `agRatio/agAreaRM1.YYYY`, `agAreaRM2.YYYY`, `agWatRM1.YYYY`, `agWatRM2.YYYY` (×28 each) | M1 (stationary) / M2 (regime-aware) counterfactual variants for the coupled MODFLOW runs | as above | **P** |
| `agRatio/agRatio.csv`, `agRatioM1.csv`, `agRatioM2.csv` | Wide annual irrigation-to-precipitation ratio by agent | dimensionless | **P** |

> **agRatio provenance.** This deposit includes the project-generated `M1`/`M2`
> variants (`agRatio_M1M2.zip`) and `agRatio*.csv`. The base `agAreaR.YYYY`/`agWatR.YYYY`
> series are verbatim RRCA-format MODFLOW inputs and are **not deposited** — obtain
> them from the RRCA MODFLOW-2000 model (**cite McKusick, 2003**).

## Units summary
irrigation/pumping depth = **mm** (also ft, in); precipitation = **mm**; temperature = **°C**;
crop prices = **US$/bushel**; diesel = **US$/gallon**; water amount = **acre-feet**;
area/irrigation ratios = **dimensionless**.

## Agent key
- **46** county-level RRCA decision units; **43** with complete 1993–2020 records are used.
- Agent IDs present: **{1–32, 36–40, 43–48}**; absent: {33, 34, 35, 41, 42}.
- **DTC clusters (k=2):** Cluster 2 (minority, high-variability) = **{2, 3, 24, 28, 29}**; Cluster 1 = the other 38.
- **Non-stationary agents (BOCPD, p ≥ 0.3) = {2, 12, 14, 18, 20, 24, 28, 29}** (8 agents). Agent 3 is in Cluster 2 but did not exceed the threshold and is excluded from predictive analysis.
- **Operational changepoints cp\*:** 2004 (12, 14, 18, 24), 2005 (20), 2011 (29), 2012 (2, 28).
- *TODO (Yao):* add the agent-ID → county-name lookup if you want county labels public (not currently in the repo).
