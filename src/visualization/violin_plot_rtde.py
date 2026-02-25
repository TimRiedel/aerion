from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def plot_rtde_violins(
    target_rtd: np.ndarray,
    pred_rtde: np.ndarray,
    bin_edges_km: list[int] = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
    violin_width: float = 0.7,
    dpi: int = 150,
    cmap: str = "viridis",
    is_relative_rtde: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create violin plot of prediction error (pred_rtd - target_rtd) by target RTD bucket.

    Parameters
    ----------
    target_rtd : np.ndarray
        Target RTD per trajectory in km, shape (B,).
    pred_rtde : np.ndarray
        Prediction error in km (pred_rtd - target_rtd) per trajectory, shape (B,).
    bin_edges_km : np.ndarray, optional
        Bin edges in km. Default: 25, 50, ..., 300 (buckets 25-50, ..., 275-300).
    violin_width : float
        Width of each violin along x-axis.
    dpi : int
        Figure DPI.
    cmap : str
        Colormap for the violins.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    bin_edges_km = np.array(bin_edges_km)
    bin_labels = [f"[{int(lo)}, {int(hi)}]" for lo, hi in zip(bin_edges_km[:-1], bin_edges_km[1:])]
    n_bins = len(bin_edges_km) - 1

    groups = group_rtd_by_target_bucket(target_rtd, pred_rtde, bin_edges_km)
    violin_positions = np.arange(1, n_bins + 1, dtype=float)
    non_empty_violin_positions = [violin_positions[i] for i in range(n_bins) if len(groups[i]) > 0] # Filter out empty positions
    non_empty_groups = [g for g in groups if len(g) > 0] # Filter out empty groups
    non_empty_indices = np.where(np.array([len(g) > 0 for g in groups]))[0]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    if not non_empty_violin_positions:
        style_axes(ax, np.arange(1, n_bins + 1, dtype=float), bin_labels, is_relative_rtde)
        return fig, ax

    parts = ax.violinplot(
        non_empty_groups,
        positions=non_empty_violin_positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    cmap = plt.get_cmap(cmap)
    for i, pc in enumerate(parts["bodies"]):
        idx = int(non_empty_indices[i])
        color = cmap(idx / max(n_bins - 1, 1))
        pc.set_facecolor(color)
        pc.set_alpha(0.9)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.2)

    add_box_plot(ax, non_empty_groups, non_empty_violin_positions, violin_width)
    style_axes(ax, violin_positions, bin_labels, is_relative_rtde)

    return fig, ax


def bucket_target_rtd(
    target_rtd: np.ndarray,
    bin_edges_km: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each trajectory to a target-RTD bucket index.

    Parameters
    ----------
    target_rtd : np.ndarray
        Target RTD per trajectory in km, shape (B,).
    bin_edges_km : np.ndarray
        Bin edges in km (e.g. 25, 50, ..., 300).

    Returns
    -------
    bucket_indices : np.ndarray
        Integer bucket index per sample (0 to n_bins-1). Samples outside
        (bin_edges[0], bin_edges[-1]] are assigned -1 and should be excluded.
    mask_valid : np.ndarray
        Boolean mask True where target_rtd is within the bin range.
    """
    # np.digitize: 1 for (edges[0], edges[1]], ..., n for (edges[-2], edges[-1]]
    bucket_indices = np.digitize(target_rtd, bin_edges_km, right=True) - 1
    mask_valid = (target_rtd > bin_edges_km[0]) & (target_rtd <= bin_edges_km[-1])
    bucket_indices = np.where(mask_valid, bucket_indices, -1)

    return bucket_indices, mask_valid


def group_rtd_by_target_bucket(
    target_rtd: np.ndarray,
    pred_rtde: np.ndarray,
    bin_edges_km: np.ndarray,
) -> List[np.ndarray]:
    """Return list of pred_rtde arrays, one per bucket (only valid buckets)."""
    bucket_indices, mask_valid = bucket_target_rtd(target_rtd, bin_edges_km)
    n_bins = len(bin_edges_km) - 1
    groups = []
    for b in range(n_bins):
        in_b = (bucket_indices == b) & mask_valid
        groups.append(pred_rtde[in_b])
    return groups


def calculate_box_whisker_limits(data: np.ndarray, q1: float, q3: float) -> Tuple[float, float]:
    """Standard box plot whiskers: extent to min/max of data within [Q1-1.5*IQR, Q3+1.5*IQR]."""
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    in_low = data[data >= lower]
    in_high = data[data <= upper]
    low_whisk = float(np.min(in_low)) if in_low.size else lower
    high_whisk = float(np.max(in_high)) if in_high.size else upper
    return low_whisk, high_whisk


def add_box_plot(
    ax: plt.Axes,
    groups: List[np.ndarray],
    positions: np.ndarray,
    width: float,
) -> None:
    """
    Add a box plot (white fill, median, whiskers) and 99% central interval lines per violin.
    """
    box_width = width * 0.3
    half = box_width / 2
    for i, data in enumerate(groups):
        if len(data) == 0:
            continue
        pos = positions[i]
        q_05, q_25, q_50, q_75, q_95 = np.percentile(data, [0.5, 25, 50, 75, 99.5])
        low_whisk, high_whisk = calculate_box_whisker_limits(data, q_25, q_75)

        # Box: IQR rectangle, white fill, black edge
        rect = Rectangle(
            (pos - half, q_25), box_width, q_75 - q_25,
            facecolor="white", edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)
        # Median line inside box
        ax.hlines(q_50, pos - half, pos + half, colors="black", linewidth=2.0)
        # Whiskers: vertical lines from box to whisker ends
        ax.plot([pos, pos], [q_25, low_whisk], color="black", linewidth=1.2)
        ax.plot([pos, pos], [q_75, high_whisk], color="black", linewidth=1.2)
        # 99% interval: gray line, same width as box
        ax.hlines(q_05, pos - half, pos + half, colors="gray", linewidth=1.2)
        ax.hlines(q_95, pos - half, pos + half, colors="gray", linewidth=1.2)
        # Whisker caps, slightly longer than box
        ax.plot([pos - half - half/2, pos + half + half/2], [low_whisk, low_whisk], color="black", linewidth=1.2)
        ax.plot([pos - half - half/2, pos + half + half/2], [high_whisk, high_whisk], color="black", linewidth=1.2)



def style_axes(
    ax: plt.Axes,
    positions: np.ndarray,
    bin_labels: List[str],
    is_relative_rtde: bool,
) -> None:
    """Set axis labels, ticks, grid and legend."""
    ax.set_xlabel("Target Distance (km)")
    if is_relative_rtde:
        ax.set_ylabel("Relative RTD Prediction Error (%)")
    else:
        ax.set_ylabel("RTD Prediction Error (km)")
    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels)

    # Fixed x-limits so empty leading/trailing bins still align with their ticks
    n = len(positions)
    ax.set_xlim(0.5, n + 0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legend for percentiles (use proxy artists)
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1.2, label=r"Tukey Whiskers ($\mathregular{1.5 \cdot IQR}$)"),
        Line2D([0], [0], color="gray", linewidth=1.2, label="99% Central Interval"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

