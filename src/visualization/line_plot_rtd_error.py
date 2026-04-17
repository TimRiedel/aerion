from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from .violin_plot_rtde import group_rtd_by_target_bucket


def plot_rtd_error_line(
    target_rtd: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    color_factor: float = 0.2,
    bin_edges_km: list[int] = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a line plot of per-bucket mean with IQR and 1.5×IQR bands for a single error metric.

    Parameters
    ----------
    target_rtd : np.ndarray
        Target RTD per trajectory in km, shape (B,).
    values : np.ndarray
        Per-trajectory error values (e.g. absolute km error or percentage error), shape (B,).
    ylabel : str
        Label for the y-axis (e.g. "RTD MAE (km)" or "RTD MAPE (%)").
    color :
        Matplotlib color for the line and std band.
    bin_edges_km : list[int]
        Bin edges in km.
    dpi : int
        Figure DPI.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    color = cm.get_cmap("viridis")(color_factor)
    bin_edges_km = np.array(bin_edges_km)
    bin_labels = [f"[{int(lo)}, {int(hi)}]" for lo, hi in zip(bin_edges_km[:-1], bin_edges_km[1:])]
    n_bins = len(bin_edges_km) - 1
    x = np.arange(n_bins)

    groups = group_rtd_by_target_bucket(target_rtd, values, bin_edges_km)
    mean_per_bucket = np.array([g.mean() if len(g) > 0 else np.nan for g in groups])
    q25 = np.array([np.percentile(g, 25) if len(g) > 0 else np.nan for g in groups])
    q75 = np.array([np.percentile(g, 75) if len(g) > 0 else np.nan for g in groups])
    iqr = q75 - q25
    whisker_lo = np.array([max(np.min(g[g >= q25[i] - 1.5 * iqr[i]]), q25[i] - 1.5 * iqr[i])
                           if len(g) > 0 else np.nan for i, g in enumerate(groups)])
    whisker_hi = np.array([min(np.max(g[g <= q75[i] + 1.5 * iqr[i]]), q75[i] + 1.5 * iqr[i])
                           if len(g) > 0 else np.nan for i, g in enumerate(groups)])

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    ax.fill_between(x, whisker_lo, whisker_hi, color=color, alpha=0.12, linewidth=0, zorder=1)
    ax.fill_between(x, q25, q75, color=color, alpha=0.25, linewidth=0, zorder=2)
    ax.plot(x, mean_per_bucket, color=color, marker="o", linewidth=2.0, zorder=3, label="Mean")

    trans = offset_copy(ax.transData, fig=fig, y=10, units="points")
    for xi, val in zip(x, mean_per_bucket):
        if not np.isnan(val):
            ax.text(xi, val, f"{val:.1f}", color=color,
                    ha="center", va="bottom", zorder=4, transform=trans)

    band_iqr = Patch(facecolor=color, alpha=0.18, label="1.5×IQR")
    band_q = Patch(facecolor=color, alpha=0.45, label="IQR (25–75th pct.)")
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines + [band_q, band_iqr], labels + [band_q.get_label(), band_iqr.get_label()], loc="upper left")

    ax.set_xlabel("Target RTD (km)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig, ax
