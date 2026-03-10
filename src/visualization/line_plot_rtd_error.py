from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from .violin_plot_rtde import group_rtd_by_target_bucket


def plot_rtd_error_line(
    target_rtd: np.ndarray,
    rtde_km: np.ndarray,
    rtdpe: np.ndarray,
    bin_edges_km: list[int] = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a dual-axis line plot of per-bucket RTD MAE (km, blue, left axis)
    and MAPE (%, orange, right axis) by target RTD bucket.

    Parameters
    ----------
    target_rtd : np.ndarray
        Target RTD per trajectory in km, shape (B,).
    rtde_km : np.ndarray
        Signed RTD prediction error in km per trajectory, shape (B,).
    rtdpe : np.ndarray
        Signed RTD percentage error in % per trajectory, shape (B,).
    bin_edges_km : list[int]
        Bin edges in km. Default matches the violin plot buckets.
    dpi : int
        Figure DPI.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_mae : matplotlib.axes.Axes  (left, MAE)
    """
    bin_edges_km = np.array(bin_edges_km)
    bin_labels = [f"[{int(lo)}, {int(hi)}]" for lo, hi in zip(bin_edges_km[:-1], bin_edges_km[1:])]
    n_bins = len(bin_edges_km) - 1
    x = np.arange(n_bins)

    mae_groups = group_rtd_by_target_bucket(target_rtd, np.abs(rtde_km), bin_edges_km)
    mape_groups = group_rtd_by_target_bucket(target_rtd, np.abs(rtdpe), bin_edges_km)

    mae_per_bucket = np.array([g.mean() if len(g) > 0 else np.nan for g in mae_groups])
    mape_per_bucket = np.array([g.mean() if len(g) > 0 else np.nan for g in mape_groups])

    fig, ax_mae = plt.subplots(figsize=(12, 6), dpi=dpi)
    ax_mape = ax_mae.twinx()

    viridis = cm.get_cmap("viridis")
    color_mae = viridis(0.2)
    color_mape = viridis(0.65)

    ax_mae.plot(x, mae_per_bucket, color=color_mae, marker="o", linewidth=2.0, label="MAE (km)")
    ax_mape.plot(x, mape_per_bucket, color=color_mape, marker="s", linewidth=2.0, label="MAPE (%)")

    ax_mae.set_xlabel("Target Distance (km)")
    ax_mae.set_ylabel("RTD MAE (km)")
    ax_mae.tick_params(axis="y")
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(bin_labels)
    ax_mae.set_xlim(-0.5, n_bins - 0.5)
    ax_mae.grid(True, linestyle="--", alpha=0.5)

    ax_mape.set_ylabel("RTD MAPE (%)")
    ax_mape.tick_params(axis="y")

    lines_mae, labels_mae = ax_mae.get_legend_handles_labels()
    lines_mape, labels_mape = ax_mape.get_legend_handles_labels()
    ax_mae.legend(lines_mae + lines_mape, labels_mae + labels_mape, loc="upper left")

    fig.tight_layout()
    return fig, ax_mae
