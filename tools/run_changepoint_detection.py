"""Run Bayesian offline changepoint detection on irrigation depth time series.

Supports two modes:
  1. **Cluster mode** (default): Loads cluster assignments from DTC and detects
     changepoints in each cluster's mean annual irrigation depth.
  2. **Per-agent mode** (``--agent-ids``): Runs BOCPD on individual agent time
     series, bypassing the cluster workflow entirely.

Usage:
    # Cluster mode (default k=2)
    python tools/run_changepoint_detection.py
    python tools/run_changepoint_detection.py --k 3
    python tools/run_changepoint_detection.py --threshold 0.5
    python tools/run_changepoint_detection.py --k 2 --year-end 2008 \
        --assignments-csv results/dtc_cluster_assignments_k2_1993-2004.csv

    # Per-agent mode
    python tools/run_changepoint_detection.py --agent-ids 4 11

Outputs:
    Cluster mode:   results/changepoint_probabilities_k{K}.csv
    Per-agent mode: results/changepoint_probabilities_agents_4_11.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.run_dtc_clustering import load_data

# --- Paths ---
RESULTS_DIR = BASE_DIR / "results"

# --- Default hyperparameters ---
DEFAULT_THRESHOLD = 0.5
DEFAULT_ALPHA0 = 1.0
DEFAULT_BETA0 = 1.0
DEFAULT_KAPPA0 = 1.0
DEFAULT_MU0 = 0.0


def load_cluster_assignments(k, assignments_path=None):
    """Load cluster assignments from CSV.

    Parameters
    ----------
    k : int
        Number of clusters (used to find default file).
    assignments_path : str or Path, optional
        Explicit path to assignments CSV. Overrides the default k-based path.
    """
    if assignments_path is not None:
        path = Path(assignments_path)
    else:
        path = RESULTS_DIR / f"dtc_cluster_assignments_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {path}\n"
            f"Run: python tools/run_dtc_clustering.py --k {k}"
        )
    return pd.read_csv(path)


def compute_cluster_means(values, years, agent_ids, assignments):
    """Compute mean annual irrigation depth per cluster.

    Returns dict mapping cluster_id -> (years_array, mean_values_array).
    """
    years_arr = np.array(years)
    cluster_means = {}
    for cluster_id in sorted(assignments["Cluster"].unique()):
        cluster_agent_ids = assignments[
            assignments["Cluster"] == cluster_id
        ]["AgentID"].values
        mask = np.isin(agent_ids, cluster_agent_ids)
        cluster_values = values[mask]
        cluster_mean = cluster_values.mean(axis=0)
        cluster_means[cluster_id] = (years_arr, cluster_mean)
    return cluster_means


def compute_agent_series(values, years, agent_ids, target_ids):
    """Extract individual agent time series.

    Parameters
    ----------
    values : np.ndarray, shape (n_agents, T)
        Irrigation depth matrix from ``load_data()``.
    years : list[int]
        Year labels.
    agent_ids : np.ndarray
        Agent IDs corresponding to rows of *values*.
    target_ids : list[int]
        Agent IDs to extract.

    Returns
    -------
    dict mapping agent_id -> (years_array, values_array)
        Same structure as ``compute_cluster_means()`` so downstream code can
        reuse it unchanged.
    """
    years_arr = np.array(years)
    agent_series = {}
    for aid in target_ids:
        mask = agent_ids == aid
        if not mask.any():
            raise ValueError(
                f"Agent {aid} not found in data. "
                f"Available: {sorted(agent_ids.tolist())}"
            )
        agent_series[aid] = (years_arr, values[mask].squeeze())
    return agent_series


def preprocess_series(years, values, rolling_window=None, filter_zeros=False):
    """Apply smoothing and/or zero-filtering to an irrigation time series.

    Order: smooth first on full series, then filter zeros based on raw values.

    Parameters
    ----------
    years : np.ndarray
        Year labels.
    values : np.ndarray
        Irrigation depth values (1D).
    rolling_window : int or None
        Centered rolling mean window size. ``min_periods=1`` handles edges.
    filter_zeros : bool
        If True, remove years where the **raw** (pre-smoothed) values are zero.

    Returns
    -------
    (filtered_years, filtered_values) : tuple of np.ndarray
    """
    raw_values = values.copy()
    result_values = values.copy().astype(float)

    # Step 1: rolling mean on full series
    if rolling_window is not None:
        result_values = (
            pd.Series(result_values)
            .rolling(rolling_window, center=True, min_periods=1)
            .mean()
            .values
        )

    # Step 2: filter zeros based on raw values
    if filter_zeros:
        nonzero_mask = raw_values > 0
        years = years[nonzero_mask]
        result_values = result_values[nonzero_mask]

    return years, result_values


class OfflineLinearRegressionLikelihood:
    """Offline Bayesian linear regression likelihood for changepoint detection.

    Models each segment as y = alpha + beta * t + eps, eps ~ N(0, sigma^2)
    with Normal-Inverse-Gamma conjugate prior. Returns log marginal likelihood
    in closed form.

    Parameters
    ----------
    a0 : float
        InvGamma shape prior on noise variance.
    b0 : float
        InvGamma scale prior on noise variance.
    Lambda0_scale : float
        Prior precision scale on regression coefficients.
        Lambda0 = Lambda0_scale * I_2.
    device : torch.device or None
    cache_enabled : bool
    """

    def __init__(self, a0=1.0, b0=1.0, Lambda0_scale=0.01,
                 device=None, cache_enabled=True):
        import torch
        from bayesian_changepoint_detection.device import get_device

        self.device = get_device(device)
        self.a0 = a0
        self.b0 = b0
        self.Lambda0 = torch.tensor(
            [[Lambda0_scale, 0.0], [0.0, Lambda0_scale]],
            dtype=torch.float64, device=self.device,
        )
        self.m0 = torch.zeros(2, dtype=torch.float64, device=self.device)
        self.cache_enabled = cache_enabled
        self._cache = {}
        self._cached_data = None

    def _check_cache(self, data, t, s):
        if not self.cache_enabled:
            return None
        import torch
        if self._cached_data is None or not torch.equal(data, self._cached_data):
            self._cache.clear()
            self._cached_data = data.clone()
            return None
        return self._cache.get((t, s), None)

    def _store_cache(self, t, s, result):
        if self.cache_enabled:
            self._cache[(t, s)] = result

    def pdf(self, data, t, s):
        """Log marginal likelihood for segment data[t:s] under linear regression.

        Returns float (matches BaseLikelihood interface).
        """
        import torch

        cached = self._check_cache(data, t, s)
        if cached is not None:
            return cached

        segment = data[t:s].to(dtype=torch.float64, device=self.device)
        n = len(segment)
        if n == 0:
            return -1e30

        # Sufficient statistics (no explicit X matrix)
        idx = torch.arange(n, dtype=torch.float64, device=self.device)
        sum_i = n * (n - 1) / 2.0
        sum_i2 = n * (n - 1) * (2 * n - 1) / 6.0
        XtX = torch.tensor(
            [[float(n), sum_i], [sum_i, sum_i2]],
            dtype=torch.float64, device=self.device,
        )
        Xty = torch.stack([segment.sum(), (idx * segment).sum()])
        yty = (segment * segment).sum()

        # Posterior
        Lambda_n = XtX + self.Lambda0
        Lambda_n_inv = torch.linalg.inv(Lambda_n)
        m_n = Lambda_n_inv @ (Xty + self.Lambda0 @ self.m0)
        a_n = self.a0 + n / 2.0
        b_n = (
            self.b0
            + 0.5 * (yty + self.m0 @ self.Lambda0 @ self.m0 - m_n @ Lambda_n @ m_n)
        )
        b_n = max(b_n.item(), 1e-12)  # floor to prevent log(0)

        # Log marginal likelihood
        _, logdet_L0 = torch.linalg.slogdet(self.Lambda0)
        _, logdet_Ln = torch.linalg.slogdet(Lambda_n)

        log_ml = (
            -n / 2.0 * np.log(2.0 * np.pi)
            + 0.5 * logdet_L0.item()
            - 0.5 * logdet_Ln.item()
            + self.a0 * np.log(self.b0)
            - a_n * np.log(b_n)
            + float(torch.lgamma(torch.tensor(a_n, dtype=torch.float64)))
            - float(torch.lgamma(torch.tensor(self.a0, dtype=torch.float64)))
        )

        self._store_cache(t, s, log_ml)
        return log_ml


class OnlineLinearRegressionLikelihood:
    """Online Bayesian linear regression likelihood for changepoint detection.

    Maintains sufficient statistics per run length. Computes predictive
    Student-t distribution for each possible run length.

    Parameters
    ----------
    a0 : float
        InvGamma shape prior on noise variance.
    b0 : float
        InvGamma scale prior on noise variance.
    Lambda0_scale : float
        Prior precision scale on regression coefficients.
    device : torch.device or None
    """

    def __init__(self, a0=0.1, b0=0.01, Lambda0_scale=0.01, device=None):
        import torch
        from bayesian_changepoint_detection.device import get_device

        self.device = get_device(device)
        self.a0 = a0
        self.b0 = b0
        self.Lambda0_scale = Lambda0_scale
        self.t = 0

        # Sufficient statistics per run length (scalars stored as 1D tensors)
        self.n_obs = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.sum_y = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.sum_ty = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.sum_yy = torch.zeros(1, dtype=torch.float64, device=self.device)

    def _compute_posterior(self):
        """Compute posterior params for all run lengths from sufficient stats."""
        import torch

        n = self.n_obs  # shape (R,)
        s0 = self.Lambda0_scale

        # X'X entries
        sum_i = n * (n - 1.0) / 2.0
        sum_i2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0

        # Lambda_n components: [[n + s0, sum_i], [sum_i, sum_i2 + s0]]
        L11 = n + s0
        L12 = sum_i
        L22 = sum_i2 + s0

        # Determinant of Lambda_n (2x2)
        det_Ln = L11 * L22 - L12 * L12

        # Inverse of Lambda_n (2x2): [[L22, -L12], [-L12, L11]] / det
        inv_L11 = L22 / det_Ln
        inv_L12 = -L12 / det_Ln
        inv_L22 = L11 / det_Ln

        # m_n = Lambda_n^{-1} @ X'y (since Lambda0 @ m0 = 0 with m0=0)
        m_n0 = inv_L11 * self.sum_y + inv_L12 * self.sum_ty
        m_n1 = inv_L12 * self.sum_y + inv_L22 * self.sum_ty

        # a_n
        a_n = self.a0 + n / 2.0

        # b_n = b0 + 0.5*(y'y - m_n' Lambda_n m_n)  [since m0=0]
        mLm = m_n0 * (L11 * m_n0 + L12 * m_n1) + m_n1 * (L12 * m_n0 + L22 * m_n1)
        b_n = self.b0 + 0.5 * (self.sum_yy - mLm)
        b_n = torch.clamp(b_n, min=1e-12)

        return a_n, b_n, m_n0, m_n1, inv_L11, inv_L12, inv_L22, det_Ln

    def pdf(self, data):
        """Predictive log probability for new observation under each run length.

        Returns torch.Tensor of shape (self.t,) with log probabilities.
        """
        import torch
        import torch.distributions as dist

        data = data.to(dtype=torch.float64, device=self.device)
        if data.numel() != 1:
            raise ValueError("Expected scalar input")
        y_new = data.squeeze()

        self.t += 1

        if self.t == 1:
            # First observation: use prior predictive
            # x_new = [1, 0], prior: m0=0, Lambda0 = s0*I, a0, b0
            s0 = self.Lambda0_scale
            # pred variance: b0/a0 * (1 + x'Lambda0^{-1}x) = b0/a0 * (1 + 1/s0)
            pred_scale = torch.tensor(
                (self.b0 / self.a0 * (1.0 + 1.0 / s0)) ** 0.5,
                dtype=torch.float64, device=self.device,
            )
            df = torch.tensor(2.0 * self.a0, dtype=torch.float64, device=self.device)
            t_dist = dist.StudentT(df=df, loc=0.0, scale=pred_scale)
            return t_dist.log_prob(y_new).unsqueeze(0).float()

        a_n, b_n, m_n0, m_n1, inv_L11, inv_L12, inv_L22, det_Ln = (
            self._compute_posterior()
        )

        # x_new = [1, n] for each run length (n = current count)
        n = self.n_obs  # shape (R,)

        # Predictive mean: x_new' @ m_n
        pred_mean = m_n0 + n * m_n1

        # Predictive variance: b_n/a_n * (1 + x_new' Lambda_n^{-1} x_new)
        # x' Linv x = inv_L11 + 2*n*inv_L12 + n^2*inv_L22
        xLinvx = inv_L11 + 2.0 * n * inv_L12 + n * n * inv_L22
        pred_var = (b_n / a_n) * (1.0 + xLinvx)
        pred_scale = torch.sqrt(torch.clamp(pred_var, min=1e-12))

        df = 2.0 * a_n

        log_probs = torch.zeros(self.t, dtype=torch.float64, device=self.device)
        for i in range(self.t):
            t_dist = dist.StudentT(df=df[i], loc=pred_mean[i], scale=pred_scale[i])
            log_probs[i] = t_dist.log_prob(y_new)

        return log_probs.float()

    def update_theta(self, data, **kwargs):
        """Update sufficient statistics with new observation."""
        import torch

        data = data.to(dtype=torch.float64, device=self.device)
        y_new = data.squeeze()
        n = self.n_obs  # current counts per run length

        # Update stats for existing run lengths
        new_sum_y = self.sum_y + y_new
        new_sum_ty = self.sum_ty + n * y_new
        new_sum_yy = self.sum_yy + y_new * y_new
        new_n = n + 1.0

        # Prepend fresh prior (run length 0)
        zero = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.n_obs = torch.cat([zero, new_n])
        self.sum_y = torch.cat([zero, new_sum_y])
        self.sum_ty = torch.cat([zero, new_sum_ty])
        self.sum_yy = torch.cat([zero, new_sum_yy])


def run_offline_slope_detection(data, prior_p=None, a0=1.0, b0=1.0,
                                Lambda0_scale=0.01):
    """Run offline BOCPD with linear regression likelihood (slope change detection).

    Parameters
    ----------
    data : np.ndarray, shape (T,)
        Raw time series (not differenced).
    prior_p : float, optional
        Prior changepoint probability per step. Default: 1/(T+1).
    a0, b0 : float
        InvGamma prior on noise variance.
    Lambda0_scale : float
        Prior precision on regression coefficients.

    Returns
    -------
    changepoint_probs : np.ndarray, shape (T-1,)
    """
    import torch
    from functools import partial
    try:
        from bayesian_changepoint_detection import offline_changepoint_detection, const_prior
    except ImportError:
        from bayesian_changepoint_detection import offline_changepoint_detection

        def const_prior(t, p, device=None):
            return torch.tensor(p, device=device)

    T = len(data)
    if prior_p is None:
        prior_p = 1.0 / (T + 1)

    device = torch.device("cpu")
    data_tensor = torch.tensor(data, dtype=torch.float64, device=device)

    prior_func = partial(const_prior, p=prior_p, device=device)
    likelihood = OfflineLinearRegressionLikelihood(
        a0=a0, b0=b0, Lambda0_scale=Lambda0_scale, device=device,
    )

    Q, P, Pcp = offline_changepoint_detection(
        data_tensor, prior_func, likelihood, device=device
    )

    changepoint_probs = torch.exp(Pcp).sum(0)
    return changepoint_probs.numpy()


def run_online_slope_detection(data, hazard_lam=None, a0=0.1, b0=0.01,
                               Lambda0_scale=0.01):
    """Run online BOCPD with linear regression likelihood (slope change detection).

    Parameters
    ----------
    data : np.ndarray, shape (T,)
        Raw time series (not differenced).
    hazard_lam : float, optional
        Expected run length for constant hazard. Default: T.
    a0, b0 : float
        InvGamma prior on noise variance.
    Lambda0_scale : float
        Prior precision on regression coefficients.

    Returns
    -------
    changepoint_probs : np.ndarray, shape (T+1,)
    R : np.ndarray, shape (T+1, T+1)
    """
    import torch
    from functools import partial
    from bayesian_changepoint_detection import online_changepoint_detection, constant_hazard

    T = len(data)
    if hazard_lam is None:
        hazard_lam = float(T)

    device = torch.device("cpu")
    data_tensor = torch.tensor(data, dtype=torch.float64, device=device)

    hazard_func = partial(constant_hazard, hazard_lam, device=device)
    likelihood = OnlineLinearRegressionLikelihood(
        a0=a0, b0=b0, Lambda0_scale=Lambda0_scale, device=device,
    )

    R, cp_probs = online_changepoint_detection(
        data_tensor, hazard_func, likelihood, device=device
    )

    return cp_probs.numpy(), R.numpy()


def run_offline_detection(data, prior_p=None, alpha0=DEFAULT_ALPHA0,
                          beta0=DEFAULT_BETA0, kappa0=DEFAULT_KAPPA0,
                          mu0=DEFAULT_MU0):
    """Run offline Bayesian changepoint detection on a 1D time series.

    Parameters
    ----------
    data : np.ndarray, shape (T,)
        Raw time series values (not normalized).
    prior_p : float, optional
        Prior probability of changepoint at each step. Default: 1/(T+1).
    alpha0, beta0, kappa0, mu0 : float
        StudentT likelihood hyperparameters (Normal-Gamma conjugate prior).

    Returns
    -------
    changepoint_probs : np.ndarray, shape (T-1,)
        Posterior probability of a changepoint at each position.
        Index t = changepoint between data[t] and data[t+1].
    """
    import torch
    from functools import partial
    try:
        from bayesian_changepoint_detection import offline_changepoint_detection, const_prior
    except ImportError:
        from bayesian_changepoint_detection import offline_changepoint_detection

        def const_prior(t, p, device=None):
            return torch.tensor(p, device=device)

    from bayesian_changepoint_detection.offline_likelihoods import StudentT

    T = len(data)
    if prior_p is None:
        prior_p = 1.0 / (T + 1)

    device = torch.device("cpu")
    data_tensor = torch.tensor(data, dtype=torch.float64, device=device)

    prior_func = partial(const_prior, p=prior_p, device=device)
    likelihood = StudentT(
        alpha0=alpha0, beta0=beta0, kappa0=kappa0, mu0=mu0, device=device
    )

    Q, P, Pcp = offline_changepoint_detection(
        data_tensor, prior_func, likelihood, device=device
    )

    changepoint_probs = torch.exp(Pcp).sum(0)
    return changepoint_probs.numpy()


def run_online_detection(data, hazard_lam=None, alpha0=0.1, beta0=0.01,
                         kappa0=DEFAULT_KAPPA0, mu0=DEFAULT_MU0):
    """Run online Bayesian changepoint detection on a 1D time series.

    Processes data sequentially — changepoint_probs[t] uses only data up
    to time t (Adams & MacKay, 2007).

    Parameters
    ----------
    data : np.ndarray, shape (T,)
        Raw time series values (not normalized).
    hazard_lam : float, optional
        Expected run length for constant hazard function.
        Default: T (= 1/T hazard probability per step).
    alpha0, beta0, kappa0, mu0 : float
        StudentT online likelihood hyperparameters (Normal-Gamma conjugate prior).
        Note: online defaults differ from offline — the online StudentT
        uses (alpha=0.1, beta=0.01) to be weakly informative for sequential
        updating, while the offline version uses (1.0, 1.0).

    Returns
    -------
    changepoint_probs : np.ndarray, shape (T+1,)
        Posterior probability of a changepoint at each time step.
        Index t corresponds to the changepoint occurring at data[t].
    R : np.ndarray, shape (T+1, T+1)
        Run length probability matrix.
    """
    import torch
    from functools import partial
    from bayesian_changepoint_detection import online_changepoint_detection, constant_hazard
    from bayesian_changepoint_detection.online_likelihoods import StudentT

    T = len(data)
    if hazard_lam is None:
        hazard_lam = float(T)

    device = torch.device("cpu")
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    hazard_func = partial(constant_hazard, hazard_lam, device=device)
    likelihood = StudentT(
        alpha=alpha0, beta=beta0, kappa=kappa0, mu=mu0, device=device
    )

    R, cp_probs = online_changepoint_detection(
        data_tensor, hazard_func, likelihood, device=device
    )

    return cp_probs.numpy(), R.numpy()


def save_results(years, series_dict, series_probs, threshold, output_csv,
                  label="Cluster", series_slope_probs=None,
                  series_combined_probs=None):
    """Save changepoint probabilities to CSV.

    Handles the case where each series may have a different subset of years
    (e.g., after zero-filtering). Uses per-series years from ``series_dict``
    and merges on Year, filling missing entries with NaN.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid in sorted(series_dict.keys()):
        sid_years, mean_vals = series_dict[sid]
        probs = series_probs[sid]
        col_prefix = f"{label}{sid}"
        # Pad with 0 at end (no changepoint after last year)
        if len(probs) == len(sid_years) - 1:
            probs_padded = np.append(probs, 0.0)
        else:
            probs_padded = probs

        sid_df = pd.DataFrame({
            "Year": sid_years,
            f"{col_prefix}_MeanIrrigation_mm": mean_vals,
            f"{col_prefix}_CP_Prob": np.round(probs_padded, 6),
        })

        # Add slope and combined columns if available
        if series_slope_probs is not None and sid in series_slope_probs:
            slope_probs = series_slope_probs[sid]
            if len(slope_probs) == len(sid_years) - 1:
                slope_padded = np.append(slope_probs, 0.0)
            else:
                slope_padded = slope_probs
            sid_df[f"{col_prefix}_CP_Prob_Slope"] = np.round(slope_padded, 6)

        if series_combined_probs is not None and sid in series_combined_probs:
            comb_probs = series_combined_probs[sid]
            if len(comb_probs) == len(sid_years) - 1:
                comb_padded = np.append(comb_probs, 0.0)
            else:
                comb_padded = comb_probs
            sid_df[f"{col_prefix}_CP_Prob_Combined"] = np.round(comb_padded, 6)
            sid_df[f"{col_prefix}_CP_Detected"] = (
                comb_padded >= threshold
            ).astype(int)
        else:
            sid_df[f"{col_prefix}_CP_Detected"] = (
                probs_padded >= threshold
            ).astype(int)

        frames.append(sid_df)

    # Merge all series on Year (outer join handles different year subsets)
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="Year", how="outer")
    df = df.sort_values("Year").reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return df


def plot_changepoint_results(years, series_dict, series_probs, threshold,
                             n_agents_per_id, output_pdf, label="Cluster",
                             preproc_label="", series_slope_probs=None):
    """Generate multi-panel figure: time series + changepoint probabilities."""
    mpl.rcParams["font.family"] = "Arial"
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    n_series = len(series_dict)

    fig, axes = plt.subplots(
        n_series, 2,
        figsize=(14, 4 * n_series),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    if n_series == 1:
        axes = axes.reshape(1, -1)

    for row_idx, sid in enumerate(sorted(series_dict.keys())):
        years_arr, mean_vals = series_dict[sid]
        probs = series_probs[sid]
        n_agents = n_agents_per_id[sid]
        color = colors[row_idx % len(colors)]

        # --- Left panel: time series ---
        ax_ts = axes[row_idx, 0]
        ax_ts.plot(years_arr, mean_vals, color=color, linewidth=2,
                   marker="o", markersize=4, label=f"{label} {sid}")

        # Mark detected changepoints
        for i in range(len(probs)):
            if probs[i] >= threshold:
                ax_ts.axvline(x=years_arr[i] + 0.5, color="red",
                              linestyle="--", linewidth=1.5, alpha=0.8)

        if label == "Agent":
            ax_ts.set_title(f"{label} {sid}", fontsize=14)
        else:
            ax_ts.set_title(f"{label} {sid} (n={n_agents} agents)", fontsize=14)
        ax_ts.set_xlabel("Year", fontsize=14)
        ylabel = "Irrigation Depth (mm)"
        if preproc_label:
            ylabel += f" ({preproc_label})"
        ax_ts.set_ylabel(ylabel, fontsize=14)
        ax_ts.set_xlim(years_arr[0] - 0.5, years_arr[-1] + 0.5)
        ax_ts.tick_params(axis="both", labelsize=12)
        ax_ts.legend(fontsize=11, loc="upper right")

        # --- Right panel: changepoint probabilities ---
        ax_cp = axes[row_idx, 1]
        bar_positions = years_arr[:-1] + 0.5 if len(probs) == len(years_arr) - 1 else years_arr
        bar_width = 0.4
        ax_cp.bar(bar_positions - bar_width / 2, probs, width=bar_width,
                  color="#ff7f0e", alpha=0.7, label="Mean/Var Shift")

        # Add slope probs as second bar color
        if series_slope_probs is not None and sid in series_slope_probs:
            slope_probs = series_slope_probs[sid]
            ax_cp.bar(bar_positions + bar_width / 2, slope_probs, width=bar_width,
                      color="#2ca02c", alpha=0.7, label="Slope Change")

        ax_cp.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5,
                      label=f"Threshold = {threshold}")
        ax_cp.set_title(f"{label} {sid} Changepoint Probabilities", fontsize=14)
        ax_cp.set_xlabel("Year", fontsize=14)
        ax_cp.set_ylabel("Posterior Probability", fontsize=14)
        ax_cp.set_ylim(0, 1.05)
        ax_cp.set_xlim(years_arr[0] - 0.5, years_arr[-1] + 0.5)
        ax_cp.tick_params(axis="both", labelsize=12)
        ax_cp.legend(fontsize=11, loc="upper right")

    fig.tight_layout(pad=2.0)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


def plot_changepoint_results_v2(years, series_dict, series_probs, threshold,
                                n_agents_per_id, output_pdf, label="Cluster",
                                preproc_label="", series_slope_probs=None):
    """Generate stacked single-panel dual-axis figure (reference style).

    Each panel: blue time series on left y-axis, orange+green probability bars
    on right y-axis, red dotted vertical line at detected changepoint(s).
    """
    from matplotlib.patches import Patch

    mpl.rcParams["font.family"] = "Arial"
    n_series = len(series_dict)

    fig, axes = plt.subplots(
        n_series, 1,
        figsize=(12, 4.5 * n_series),
        squeeze=False,
    )

    for row_idx, sid in enumerate(sorted(series_dict.keys())):
        years_arr, mean_vals = series_dict[sid]
        probs = series_probs[sid]
        n_agents = n_agents_per_id[sid]

        # Pad level probs to match years length (prepend 0)
        if len(probs) == len(years_arr) - 1:
            probs_padded = np.insert(probs, 0, 0.0)
        else:
            probs_padded = probs

        # Pad slope probs similarly
        has_slope = (series_slope_probs is not None and sid in series_slope_probs)
        if has_slope:
            slope_probs = series_slope_probs[sid]
            if len(slope_probs) == len(years_arr) - 1:
                slope_padded = np.insert(slope_probs, 0, 0.0)
            else:
                slope_padded = slope_probs

        ax_left = axes[row_idx, 0]

        # Left y-axis: Irrigation Depth (blue)
        irr_label = "Annual Irrigation Depth (mm)"
        if preproc_label:
            irr_label += f" ({preproc_label})"
        line_irr, = ax_left.plot(
            years_arr, mean_vals, color="#1f77b4", linewidth=2,
            marker="o", markersize=5, label=irr_label,
        )
        ax_left.set_xlabel("Year", fontsize=14, fontweight="bold")
        ax_left.set_ylabel(irr_label, fontsize=14, fontweight="bold")
        ax_left.set_xlim(years_arr[0] - 0.5, years_arr[-1] + 0.5)
        ax_left.tick_params(axis="both", labelsize=12)
        ax_left.tick_params(axis="x", rotation=45)

        # Right y-axis: Posterior Probability bars
        ax_right = ax_left.twinx()
        bar_width = 0.4 if has_slope else 0.8
        offset = bar_width / 2 if has_slope else 0.0
        ax_right.bar(
            years_arr - offset, probs_padded, width=bar_width,
            color="#ff7f0e", alpha=0.7, label="Mean/Var Shift",
        )
        if has_slope:
            ax_right.bar(
                years_arr + offset, slope_padded, width=bar_width,
                color="#2ca02c", alpha=0.7, label="Slope Change",
            )
        ax_right.set_ylabel("Posterior Probability",
                            fontsize=14, fontweight="bold")
        ax_right.set_ylim(0.0, 1.0)
        ax_right.tick_params(axis="y", labelsize=12)

        # Mark the max posterior probability bar (levels)
        max_idx = np.argmax(probs_padded)
        ax_right.annotate(
            f"{probs_padded[max_idx]:.2f}",
            xy=(years_arr[max_idx] - offset, probs_padded[max_idx]),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#ff7f0e",
        )

        # Threshold line on right y-axis
        thresh_line = ax_right.axhline(
            y=threshold, color="red", linestyle="--", linewidth=1.5,
        )

        # Red dotted vertical lines at detected changepoints
        cp_line = None
        for i in range(len(series_probs[sid])):
            if series_probs[sid][i] >= threshold:
                cp_line = ax_left.axvline(
                    x=years_arr[i] + 0.5, color="red",
                    linestyle=":", linewidth=1.5,
                    label="Irrigation Nonstationarity",
                )

        # Legend (use Patch for bar representation)
        bar_patch_level = Patch(facecolor="#ff7f0e", alpha=0.7)
        handles = [line_irr, bar_patch_level]
        labels = [irr_label, "Mean/Var Shift Prob"]
        if has_slope:
            bar_patch_slope = Patch(facecolor="#2ca02c", alpha=0.7)
            handles.append(bar_patch_slope)
            labels.append("Slope Change Prob")
        handles.append(thresh_line)
        labels.append(f"Threshold = {threshold}")
        if cp_line is not None:
            handles.append(cp_line)
            labels.append("Irrigation Nonstationarity")
        ax_left.legend(handles, labels, loc="upper left", fontsize=11,
                       frameon=False, bbox_to_anchor=(0.0, 1.01))

        # Ensure line is drawn on top of bars
        ax_left.set_zorder(ax_right.get_zorder() + 1)
        ax_left.patch.set_visible(False)

        if label == "Agent":
            ax_left.set_title(f"{label} {sid}",
                              fontsize=14, fontweight="bold")
        else:
            ax_left.set_title(f"{label} {sid} (n={n_agents} agents)",
                              fontsize=14, fontweight="bold")

    fig.tight_layout(pad=2.0)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian offline changepoint detection on cluster mean irrigation"
    )
    parser.add_argument("--k", type=int, default=2,
                        help="Number of clusters (default: 2)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Probability threshold for detection (default: 0.5)")
    parser.add_argument("--prior-p", type=float, default=None,
                        help="Prior changepoint probability per step (default: 1/(T+1))")
    parser.add_argument("--alpha0", type=float, default=DEFAULT_ALPHA0)
    parser.add_argument("--beta0", type=float, default=DEFAULT_BETA0)
    parser.add_argument("--kappa0", type=float, default=DEFAULT_KAPPA0)
    parser.add_argument("--mu0", type=float, default=DEFAULT_MU0)
    parser.add_argument("--year-start", type=int, default=None,
                        help="First year to include (default: first year in data)")
    parser.add_argument("--year-end", type=int, default=None,
                        help="Last year to include (default: last year in data)")
    parser.add_argument("--assignments-csv", type=str, default=None,
                        help="Path to cluster assignments CSV (overrides default k-based path)")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online"],
                        help="Detection mode: offline (Fearnhead 2006) or online (Adams & MacKay 2007)")
    parser.add_argument("--hazard-lam", type=float, default=None,
                        help="Expected run length for online constant hazard (default: T)")
    parser.add_argument("--agent-ids", type=int, nargs="+", default=None,
                        help="Run BOCPD on individual agent time series "
                             "(bypasses cluster workflow). E.g. --agent-ids 4 11")
    parser.add_argument("--filter-zeros", action="store_true",
                        help="Exclude years with zero irrigation before BOCPD "
                             "(removes operational shutdowns)")
    parser.add_argument("--rolling-window", type=int, default=None,
                        help="Centered rolling mean window size (e.g. 3). "
                             "Applied before zero-filtering.")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Run per-agent BOCPD on all agents in this cluster "
                             "(requires --k for assignments lookup)")
    args = parser.parse_args()

    k = args.k
    per_agent = args.agent_ids is not None

    # 1. Load data (with optional year filtering)
    agent_ids, years, values = load_data(
        year_start=args.year_start, year_end=args.year_end
    )
    year_tag = f"_{years[0]}-{years[-1]}" if (args.year_start or args.year_end) else ""
    mode_tag = f"_{args.mode}" if args.mode != "offline" else ""

    print(f"Loaded {len(agent_ids)} agents, {len(years)} years "
          f"({years[0]}-{years[-1]})")

    # Build preprocessing filename tags
    preproc_tag = ""
    if args.rolling_window is not None:
        preproc_tag += f"_rm{args.rolling_window}"
    if args.filter_zeros:
        preproc_tag += "_filt"
    thresh_tag = f"_t{args.threshold}"

    # Handle --cluster: resolve agent IDs from cluster assignments
    if args.cluster is not None:
        assignments = load_cluster_assignments(k, assignments_path=args.assignments_csv)
        cluster_agents = sorted(
            assignments[assignments["Cluster"] == args.cluster]["AgentID"].tolist()
        )
        if not cluster_agents:
            raise ValueError(
                f"No agents found in cluster {args.cluster} (k={k}). "
                f"Available clusters: {sorted(assignments['Cluster'].unique().tolist())}"
            )
        args.agent_ids = cluster_agents
        per_agent = True
        print(f"Cluster {args.cluster} (k={k}): {len(cluster_agents)} agents — {cluster_agents}")

    if per_agent:
        # --- Per-agent workflow ---
        if args.cluster is not None:
            ids_tag = f"cluster{args.cluster}"
        else:
            ids_tag = "_".join(str(a) for a in sorted(args.agent_ids))
        output_csv = RESULTS_DIR / f"changepoint_probabilities_agents_{ids_tag}{thresh_tag}{preproc_tag}{mode_tag}{year_tag}.csv"
        output_pdf = RESULTS_DIR / f"changepoint_detection_agents_{ids_tag}{thresh_tag}{preproc_tag}{mode_tag}{year_tag}.pdf"
        output_pdf_v2 = RESULTS_DIR / f"changepoint_detection_v2_agents_{ids_tag}{thresh_tag}{preproc_tag}{mode_tag}{year_tag}.pdf"

        series_dict = compute_agent_series(values, years, agent_ids, args.agent_ids)

        # Apply preprocessing per agent
        for aid in sorted(series_dict.keys()):
            yr, vals = series_dict[aid]
            orig_len = len(yr)
            yr_proc, vals_proc = preprocess_series(
                yr, vals,
                rolling_window=args.rolling_window,
                filter_zeros=args.filter_zeros,
            )
            series_dict[aid] = (yr_proc, vals_proc)
            parts = []
            if args.filter_zeros:
                n_dropped = orig_len - len(yr_proc)
                parts.append(f"filtered {n_dropped} zero years")
            if args.rolling_window is not None:
                parts.append(f"applied {args.rolling_window}-yr rolling mean")
            if parts:
                print(f"  Agent {aid}: {', '.join(parts)}")

        n_agents_per_id = {aid: 1 for aid in series_dict}
        for aid in sorted(series_dict.keys()):
            _, vals = series_dict[aid]
            print(f"  Agent {aid}: range [{vals.min():.1f}, {vals.max():.1f}] mm")
        entity_label = "Agent"
    else:
        # --- Cluster workflow (original) ---
        output_csv = RESULTS_DIR / f"changepoint_probabilities_k{k}{thresh_tag}{mode_tag}{year_tag}.csv"
        output_pdf = RESULTS_DIR / f"changepoint_detection_k{k}{thresh_tag}{mode_tag}{year_tag}.pdf"
        output_pdf_v2 = RESULTS_DIR / f"changepoint_detection_v2_k{k}{thresh_tag}{mode_tag}{year_tag}.pdf"

        assignments = load_cluster_assignments(k, assignments_path=args.assignments_csv)
        print(f"Loaded cluster assignments for k={k}"
              f"{f' from {args.assignments_csv}' if args.assignments_csv else ''}")

        series_dict = compute_cluster_means(values, years, agent_ids, assignments)
        n_agents_per_id = {}
        for cid in sorted(series_dict.keys()):
            n_agents = (assignments["Cluster"] == cid).sum()
            n_agents_per_id[cid] = n_agents
            _, mean_vals = series_dict[cid]
            print(f"  Cluster {cid}: {n_agents} agents, "
                  f"mean range [{mean_vals.min():.1f}, {mean_vals.max():.1f}] mm")
        entity_label = "Cluster"

    # 4. Run changepoint detection per series (levels + slope via linear regression)
    series_probs = {}
    series_slope_probs = {}
    series_combined_probs = {}
    for sid in sorted(series_dict.keys()):
        sid_years, vals = series_dict[sid]
        print(f"\nRunning {args.mode} changepoint detection on {entity_label} {sid}...")

        # --- Levels-based detection (StudentT) ---
        if args.mode == "offline":
            level_probs = run_offline_detection(
                vals,
                prior_p=args.prior_p,
                alpha0=args.alpha0,
                beta0=args.beta0,
                kappa0=args.kappa0,
                mu0=args.mu0,
            )
        else:
            cp_probs, R = run_online_detection(
                vals,
                hazard_lam=args.hazard_lam,
                alpha0=args.alpha0,
                beta0=args.beta0,
                kappa0=args.kappa0,
                mu0=args.mu0,
            )
            level_probs = cp_probs[1:][:-1]  # shape (T-1,)
        series_probs[sid] = level_probs

        # --- Slope-based detection (linear regression likelihood) ---
        if args.mode == "offline":
            slope_probs = run_offline_slope_detection(
                vals, prior_p=args.prior_p,
                a0=args.alpha0, b0=args.beta0,
            )
        else:
            slope_cp_probs, _ = run_online_slope_detection(
                vals, hazard_lam=args.hazard_lam,
                a0=args.alpha0, b0=args.beta0,
            )
            slope_probs = slope_cp_probs[1:][:-1]  # shape (T-1,)
        series_slope_probs[sid] = slope_probs  # same shape as level_probs

        # Combined probability: max(level, slope) — no alignment needed
        combined = np.maximum(level_probs, slope_probs)
        series_combined_probs[sid] = combined

        # --- Console output ---
        detected_levels = [
            (sid_years[i], sid_years[i + 1], level_probs[i])
            for i in range(len(level_probs))
            if level_probs[i] >= args.threshold
        ]
        if detected_levels:
            print(f"  Detected changepoints (levels, threshold={args.threshold}):")
            for y1, y2, p in detected_levels:
                print(f"    Between {y1} and {y2}: p={p:.4f}")
        else:
            print(f"  No level changepoints detected above threshold {args.threshold}")

        detected_slope = [
            (sid_years[i], sid_years[i + 1], slope_probs[i])
            for i in range(len(slope_probs))
            if slope_probs[i] >= args.threshold
        ]
        if detected_slope:
            print(f"  Detected changepoints (slope change, threshold={args.threshold}):")
            for y1, y2, p in detected_slope:
                print(f"    Between {y1} and {y2}: p={p:.4f}")
        else:
            print(f"  No slope changepoints detected above threshold {args.threshold}")

    # Build preprocessing label for plots
    preproc_parts = []
    if args.rolling_window is not None:
        preproc_parts.append("smoothed")
    if args.filter_zeros:
        preproc_parts.append("filtered")
    preproc_label = ", ".join(preproc_parts)

    # 5. Save CSV
    save_results(years, series_dict, series_probs, args.threshold, output_csv,
                 label=entity_label, series_slope_probs=series_slope_probs,
                 series_combined_probs=series_combined_probs)

    # 6. Generate figures
    plot_changepoint_results(
        years, series_dict, series_probs, args.threshold,
        n_agents_per_id, output_pdf, label=entity_label,
        preproc_label=preproc_label, series_slope_probs=series_slope_probs,
    )
    plot_changepoint_results_v2(
        years, series_dict, series_probs, args.threshold,
        n_agents_per_id, output_pdf_v2, label=entity_label,
        preproc_label=preproc_label, series_slope_probs=series_slope_probs,
    )


if __name__ == "__main__":
    main()
