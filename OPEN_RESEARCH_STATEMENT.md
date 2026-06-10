# Open Research — ready-to-paste statement + citations

The live `manuscript/main.tex` Open Research section is currently `% TODO`, and
`references.bib` has no data/software entries. Paste the block below into the
`\section*{Open Research}` and add the BibTeX to `references.bib`.

> **Before pasting:** (1) replace the new code DOI placeholder once the Zenodo release exists;
> (2) add the HydroShare DOI once the resource is formally published; (3) confirm the GHCNd
> subset/version and the USDA/USEIA series + access dates.

---

## LaTeX (Open Research section)

```latex
\section*{Open Research}
The analysis and figure-generation code for this study is openly available under the
MIT License (no registration required) in Zenodo \citep{hu2026code} and is developed at
\url{https://github.com/watersecurity/gw-pumping-regime-shifts}. The processed
input data---county-level monthly groundwater irrigation depth, crop prices, precipitation,
and temperature for the High Plains Aquifer Hydrologic Observatory Area (1993--2020)---are
openly available in HydroShare under a CC-BY 4.0 license \citep{hu2026data}. Source data are
third party: daily precipitation and temperature are from the Global Historical Climatology
Network--Daily \citep{ghcnd2012}; crop commodity prices are from the U.S. Department of
Agriculture National Agricultural Statistics Service \citep{usda_nass}; diesel fuel prices
are from the U.S. Energy Information Administration \citep{useia_diesel}; and target
groundwater pumping depth is derived from the Republican River Compact Administration
MODFLOW-2000 model \citep{mckusick2003}. The deep temporal clustering implementation used in
the workflow is by \citet{forest2018dtc}.
```

## BibTeX (add to references.bib)

```bibtex
@software{hu2026code,
  author    = {Hu, Yao and Xi, Shihao},
  title     = {Non-stationary groundwater pumping behavior and coupled groundwater
               forecast uncertainty},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.1},
  doi       = {10.5281/zenodo.20618814},
  note      = {[Software], MIT License}
}

@misc{hu2026data,
  author       = {Hu, Yao},
  title        = {HPA-HOA Groundwater Pumping Behavior Data},
  year         = {2026},
  howpublished = {HydroShare},
  url          = {https://www.hydroshare.org/resource/d314b7e633024ee58649414468ad77f8},
  note         = {[Dataset], CC-BY 4.0}
}

@misc{ghcnd2012,
  author       = {Menne, Matthew J. and Durre, Imke and Korzeniewski, Bryant and others},
  title        = {Global Historical Climatology Network - Daily (GHCN-Daily), Version 3},
  year         = {2012},
  publisher    = {NOAA National Centers for Environmental Information},
  doi          = {10.7289/V5D21VHZ},    % TODO: confirm subset/access date
  note         = {[Dataset]}
}

@misc{usda_nass,
  author       = {{U.S. Department of Agriculture, National Agricultural Statistics Service}},
  title        = {Quick Stats Database: commodity prices (corn, soybean, sorghum, wheat)},
  year         = {2024},
  url          = {https://quickstats.nass.usda.gov/},
  note         = {[Dataset]. TODO: access date}
}

@misc{useia_diesel,
  author       = {{U.S. Energy Information Administration}},
  title        = {Weekly Retail On-Highway Diesel Prices},
  year         = {2024},
  url          = {https://www.eia.gov/petroleum/gasdiesel/},
  note         = {[Dataset]. TODO: access date}
}

@misc{forest2018dtc,
  author       = {Forest, Florent},
  title        = {Deep Temporal Clustering (DTC) implementation},
  year         = {2018},
  howpublished = {GitHub},
  url          = {https://github.com/FlorentF9/DeepTemporalClustering},
  note         = {[Software], MIT License}
}
```

> `mckusick2003` (RRCA model) is already cited in the manuscript. Confirm `xgboost`
> (Chen \& Guestrin, 2016), `optuna` (Akiba et al., 2019), and the BOCPD method
> (Adams \& MacKay, 2007) appear in the references; best practice is for software to carry a
> `[Software]` descriptor where applicable.

## Note on the citation-year change
The previous draft cited the code as **Hu (2024a)** pointing to Zenodo `10.5281/zenodo.15133134`,
which holds an **earlier methodology**. This statement replaces it with a **new** software
citation (`hu2026code`) for the deposit that actually reproduces this manuscript. Update any
in-text reference from "Hu (2024a)" accordingly. The HydroShare **data** citation (`hu2024data`)
is unchanged except for adding its DOI.
