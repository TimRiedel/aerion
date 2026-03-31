import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from traffic.data import airports


def plot_predictions_targets(
    input_traj: np.ndarray,
    target_traj: np.ndarray,
    pred_traj: np.ndarray,
    target_valid_len: int,
    pred_valid_len: int,
    icao: str,
    flight_id: str = None,
    target_rtd: float = None,
    pred_rtd: float = None,
    cmap: str = "viridis",
    show_waypoints: bool = True,
):
    """
    Plot prediction vs groundtruth in X/Y space (km) with altitude profile below.

    Parameters:
    -----------
    input_traj : np.ndarray
        Input trajectory [input_seq_len, 3] (x, y, altitude in meters)
    target_traj : np.ndarray
        Target trajectory [horizon_seq_len, 3] (x, y, altitude in meters)
    pred_traj : np.ndarray
        Predicted trajectory [horizon_seq_len, 3] (x, y, altitude in meters)
    target_valid_len : int
        Number of valid target waypoints
    pred_valid_len : int
        Number of valid prediction waypoints
    icao : str
        ICAO airport code for plotting runways
    flight_id : str
        Flight ID
    target_rtd : float
        Target RTD in meters
    pred_rtd : float
        Predicted RTD in meters
    cmap : str
        Colormap to use for the colors

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : tuple
        Tuple of (ax_xy, ax_altitude) axes
    """
    linewidth = 1.5
    dpi = 120

    target_traj = target_traj[:target_valid_len]
    pred_traj = pred_traj[:pred_valid_len]
    
    # Convert to km for x/y plotting, altitude stays in meters
    input_x_km, input_y_km = input_traj[:, 0] / 1000, input_traj[:, 1] / 1000
    target_x_km, target_y_km = target_traj[:, 0] / 1000, target_traj[:, 1] / 1000
    pred_x_km, pred_y_km = pred_traj[:, 0] / 1000, pred_traj[:, 1] / 1000
    input_alt, target_alt, pred_alt = input_traj[:, 2], target_traj[:, 2], pred_traj[:, 2]
    # Create figure with GridSpec for precise control over subplot alignment
    fig = plt.figure(figsize=(10, 12), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.2)
    ax_xy = fig.add_subplot(gs[0])
    ax_alt = fig.add_subplot(gs[1])

    # Reduce outer whitespace and reserve a bit of space at the top for titles.
    # This keeps both subplots aligned while shrinking the overall margins.

    # Plot airport runways in X/Y space
    ax_xy = add_airport_runways_xy(ax_xy, icao)
    
    # Plot trajectories - connect target/prediction to last input point by prepending the last input point
    target_x_connected = np.concatenate([[input_x_km[-1]], target_x_km])
    target_y_connected = np.concatenate([[input_y_km[-1]], target_y_km])
    pred_x_connected = np.concatenate([[input_x_km[-1]], pred_x_km])
    pred_y_connected = np.concatenate([[input_y_km[-1]], pred_y_km])

    # Get colors from colormap
    factors = [0.2, 0.55]
    cmap = matplotlib.colormaps[cmap]
    input_color = cmap(factors[0])
    pred_color = cmap(factors[1])
    target_color = "gold"

    ax_xy.plot(input_x_km, input_y_km, color=input_color, linewidth=linewidth, label='Input', zorder=3)
    ax_xy.plot(target_x_connected, target_y_connected, color=target_color, linewidth=linewidth, label='Target', zorder=3)
    ax_xy.plot(pred_x_connected, pred_y_connected, color=pred_color, linewidth=linewidth, label='Prediction', zorder=3)

    if show_waypoints:
        dot_size = (linewidth + 1.5) ** 2
        ax_xy.scatter(input_x_km, input_y_km, color=input_color, s=dot_size, zorder=4)
        ax_xy.scatter(target_x_connected, target_y_connected, color=target_color, s=dot_size, zorder=4)
        ax_xy.scatter(pred_x_connected, pred_y_connected, color=pred_color, s=dot_size, zorder=4)

    # Draw dashed line from the last predicted point to the last target point (runway threshold),
    # making the implied remaining distance included in pred_rtd visible in the plot.
    threshold_x_km = target_x_km[-1]
    threshold_y_km = target_y_km[-1]
    ax_xy.plot(
        [pred_x_km[-1], threshold_x_km],
        [pred_y_km[-1], threshold_y_km],
        color=pred_color, linewidth=linewidth, linestyle='--', zorder=3,
    )

    all_x_km = np.concatenate([input_x_km, target_x_km, pred_x_km])
    all_y_km = np.concatenate([input_y_km, target_y_km, pred_y_km])
    bounds_xy_km = compute_xy_bounds(all_x_km, all_y_km)
    ax_xy = set_xy_axis_config(ax_xy, bounds_xy_km)

    # Plot altitude profile
    input_len = len(input_alt)
    input_steps = np.arange(-input_len + 1, 1)
    target_steps = np.arange(0, len(target_alt) + 1)  # Include connection point at 0
    pred_steps = np.arange(0, len(pred_alt) + 1)      # Include connection point at 0
    target_alt_connected = np.concatenate([[input_alt[-1]], target_alt])
    pred_alt_connected = np.concatenate([[input_alt[-1]], pred_alt])

    ax_alt.plot(input_steps, input_alt, color=input_color, linewidth=linewidth, label='Input')
    ax_alt.plot(target_steps, target_alt_connected, color=target_color, linewidth=linewidth, label='Target')
    ax_alt.plot(pred_steps, pred_alt_connected, color=pred_color, linewidth=linewidth, label='Prediction')

    if show_waypoints:
        dot_size = (linewidth + 1.5) ** 2
        ax_alt.scatter(input_steps, input_alt, color=input_color, s=dot_size, zorder=4)
        ax_alt.scatter(target_steps, target_alt_connected, color=target_color, s=dot_size, zorder=4)
        ax_alt.scatter(pred_steps, pred_alt_connected, color=pred_color, s=dot_size, zorder=4)

    horizon_len = max(target_valid_len, pred_valid_len)
    ax_alt = set_altitude_axis_config(ax_alt, input_len, horizon_len)
    
    # Align the x-axis (left/right edges) of both plots
    pos_xy = ax_xy.get_position()
    pos_alt = ax_alt.get_position()
    ax_alt.set_position([pos_xy.x0, pos_alt.y0, pos_xy.width, pos_alt.height])
    set_title(fig, ax_xy, icao, flight_id, target_rtd, pred_rtd)

    return fig, (ax_xy, ax_alt)

def set_xy_axis_config(ax, bounds_xy_km):
    ax.set_xlim(bounds_xy_km[0], bounds_xy_km[1])
    ax.set_ylim(bounds_xy_km[2], bounds_xy_km[3])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    return ax

def set_altitude_axis_config(ax, input_len, horizon_len):
    ax.set_xlim(-input_len + 1, horizon_len)
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Altitude (m)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.7)
    return ax

def set_title(fig, ax_ref, icao, flight_id, target_rtd=None, pred_rtd=None):

    main_title = "Prediction vs Target"  # default title
    subtitle = ""
    if flight_id:
        splits = flight_id.split('_')
        callsign = splits[0]
        date = splits[2]
        date = pd.to_datetime(date).strftime("%d %b, %Y")
        flight_id_label = f'Flight: {callsign}'
        main_title = flight_id_label

        subtitle = ""
        if icao:
            subtitle += f'{icao} on '
        subtitle += f'{date}'

    # Use the reference axes to align text with the actual plotting area
    bbox = ax_ref.get_position()
    left_x = bbox.x0
    right_x = bbox.x0 + bbox.width

    # First line: left = main title (Flight ID or fallback)
    y_top = 0.93
    line_spacing = 0.025

    # Left side main title (larger), aligned with left edge of axes
    fig.text(
        left_x,
        y_top,
        main_title,
        ha="left",
        va="top",
        fontsize=16,
    )

    # Second line: left = airport/date, right = RTD label (both grey, smaller)
    if subtitle:
        fig.text(
            left_x,
            y_top - line_spacing,
            subtitle,
            ha="left",
            va="top",
            fontsize=11,
            color="grey",
            style="italic",
        )

    if target_rtd is not None and pred_rtd is not None:
        rtd_label = (
            f"Target RTD: {target_rtd / 1000:.1f} km | "
            f"Predicted RTD: {pred_rtd / 1000:.1f} km"
        )
        fig.text(
            right_x,
            y_top - line_spacing,
            rtd_label,
            ha="right",
            va="top",
            fontsize=11,
            color="grey",
            style="italic",
        )




def compute_xy_bounds(x_km, y_km, buffer_percent=0.1):
    """Compute bounds for X/Y plot with buffer."""
    min_x, max_x = x_km.min(), x_km.max()
    min_y, max_y = y_km.min(), y_km.max()
    
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    max_range = max(x_range, y_range)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    half_range = max_range / 2 * (1 + buffer_percent)
    
    return (center_x - half_range, center_x + half_range, center_y - half_range, center_y + half_range)

def get_transformer_wgs84_to_aeqd(lat, lon):
    return pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"),
        pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84"),
        always_xy=True
    )

def add_airport_runways_xy(ax, icao):
    """Add airport runways to plot in X/Y space (km). """
    
    runways = airports[icao].runways
    if runways is None:
        return ax
    
    runway_width_m = 45
    lat, lon = airports[icao].latlon
    wgs84_to_aeqd = get_transformer_wgs84_to_aeqd(lat, lon)

    for threshold_pair in runways._runways:
        if len(threshold_pair) >= 2:
            thr1, thr2 = threshold_pair[0], threshold_pair[1]
            
            x1, y1 = wgs84_to_aeqd.transform(thr1.longitude, thr1.latitude)
            x2, y2 = wgs84_to_aeqd.transform(thr2.longitude, thr2.latitude)
            
            # Calculate perpendicular direction for runway width
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Perpendicular unit vector
                perp_x, perp_y = -dy / length, dx / length
                half_width = runway_width_m / 2
                
                # Create rectangle corners
                corners = [
                    (x1 + perp_x * half_width, y1 + perp_y * half_width),
                    (x1 - perp_x * half_width, y1 - perp_y * half_width),
                    (x2 - perp_x * half_width, y2 - perp_y * half_width),
                    (x2 + perp_x * half_width, y2 + perp_y * half_width),
                ]
                
                # Convert to km and fill
                x_km = [c[0] / 1000 for c in corners]
                y_km = [c[1] / 1000 for c in corners]
                ax.fill(x_km, y_km, color='black', zorder=2)
    
    return ax