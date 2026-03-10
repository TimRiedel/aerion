from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Patch

from .violin_plot_rtde import group_rtd_by_target_bucket


def plot_rtd_error_line(
    target_rtd: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    color_factor: float = 0.2,
    bin_edges_km: list[int] = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a line plot of per-bucket mean ± 1σ for a single error metric.

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
    std_per_bucket = np.array([g.std() if len(g) > 1 else np.nan for g in groups])

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    ax.fill_between(x, mean_per_bucket - std_per_bucket, mean_per_bucket + std_per_bucket,
                    color=color, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(x, mean_per_bucket, color=color, marker="o", linewidth=2.0, zorder=2, label="Mean")

    band = Patch(facecolor=color, alpha=0.35, label="± 1σ")
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines + [band], labels + [band.get_label()], loc="upper left")

    ax.set_xlabel("Target Distance (km)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig, ax
