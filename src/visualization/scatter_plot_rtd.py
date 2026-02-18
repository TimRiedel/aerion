import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def plot_rtd_scatter(
    target_rtd: np.ndarray,
    pred_rtd: np.ndarray,
    rtde_relative: np.ndarray,
    x_max: float = 300.0,
    relative_error_min: float = -50.0,
    relative_error_max: float = 50.0,
    dpi: int = 150,
    figsize: Tuple[float, float] = (8, 6),
    point_size: float = 1.0,
    alpha: float = 1.0,
    cmap: str = "viridis",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of target vs. predicted remaining track distance (RTD).

    Parameters
    ----------
    target_rtd : np.ndarray
        Target RTD per trajectory in km, shape (B,).
    pred_rtd : np.ndarray
        Predicted RTD per trajectory in km, shape (B,).
    rtde_relative : np.ndarray
        Relative RTD error per trajectory (dimensionless, e.g. fraction),
        shape (B,). Points are colored by this value using the viridis
        colormap and a colorbar is added.
    x_max: float
        Maximum value for the x-axis.
    relative_error_min : float
        Minimum relative error to display in the colorbar.
    relative_error_max : float
        Maximum relative error to display in the colorbar.
    dpi : int
        Figure DPI.
    figsize : tuple of float
        Figure size in inches (width, height).
    point_size : float
        Marker size for individual trajectories.
    alpha : float
        Marker transparency.
    cmap : str
        Colormap for the colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    target_rtd = np.asarray(target_rtd).reshape(-1)
    pred_rtd = np.asarray(pred_rtd).reshape(-1)
    rtde_relative = np.asarray(rtde_relative).reshape(-1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    scatter_kwargs = dict(
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )

    # Limit color range to within [-100, 100] based on data
    rel_min = float(np.min(rtde_relative)) if rtde_relative.size else 0.0
    rel_max = float(np.max(rtde_relative)) if rtde_relative.size else 0.0
    vmin = max(rel_min, relative_error_min)
    vmax = min(rel_max, relative_error_max)

    sc = ax.scatter(
        target_rtd,
        pred_rtd,
        c=rtde_relative,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **scatter_kwargs,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Relative RTD error (%)")

    # Add label via a proxy artist so the legend stays compact even with colorbar
    ax.scatter(
        [],
        [],
        color="gray",
        s=point_size,
        alpha=alpha,
        label="Trajectories",
    )

    add_identity_line(ax, target_rtd, pred_rtd, x_max)
    style_scatter_axes(ax)

    return fig, ax


def add_identity_line(
    ax: plt.Axes,
    target_rtd: np.ndarray,
    pred_rtd: np.ndarray,
    x_max: float,
    color: str = "black",
    linestyle: str = "--",
    linewidth: float = 1.5,
) -> None:
    """
    Add y = x identity line and set axis bounds from individual axes.
    """
    # X-bounds from target RTD, Y-bounds from predicted RTD
    x_min = 0.0
    x_max = x_max
    y_min = 0.0
    y_max = float(np.max(pred_rtd))

    # Identity line should cover the visible square enclosing both axes
    line_lo = min(x_min, y_min)
    line_hi = max(x_max, y_max)
    ax.plot(
        [line_lo, line_hi],
        [line_lo, line_hi],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label="Ideal (Prediction = Target)",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def style_scatter_axes(ax: plt.Axes) -> None:
    """
    Apply consistent styling to the RTD scatter axes.
    """
    ax.set_xlabel("Target RTD (km)")
    ax.set_ylabel("Predicted RTD (km)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

