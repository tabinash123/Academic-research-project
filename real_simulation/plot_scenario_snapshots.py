"""Visualizer for Extracted Pedestrian Scenario Snapshots.

Loads scenario_snapshots.npz and plots high-resolution 2D bird's-eye views
for each snapshot encounter, showing:
- Asphalt road and sidewalks
- Ego vehicle position and dimensions
- 12m LiDAR perception circle
- Pedestrian live positions
- Nominal constant-velocity projections
- The full fan of 100 anchored uncertainty scenarios
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

# Setup paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))

from extract_scenario_snapshots import load_scenario_snapshots


def visualize_snapshots(
    npz_path=None,
    output_image="scenario_snapshots_visualization.png",
    show_plot=False
):
    """Generates a multi-panel visual graph of all extracted snapshots."""
    if npz_path is None:
        npz_path = current_dir / "scenario_snapshots.npz"
    else:
        npz_path = Path(npz_path)

    snapshots = load_scenario_snapshots(npz_path)
    num_snaps = len(snapshots)

    # Road & Environment Dimensions
    ROAD_Y = 20.0
    ROAD_WIDTH = 12.0
    ROAD_BOTTOM = ROAD_Y - ROAD_WIDTH / 2.0  # 14.0m
    ROAD_TOP = ROAD_Y + ROAD_WIDTH / 2.0     # 26.0m
    WORLD_HEIGHT = 40.0
    CAR_LENGTH = 3.0
    CAR_WIDTH = 2.0
    LIDAR_R = 12.0

    # Color Palette for clean scientific presentation
    COLOR_ASPHALT = "#334155"      # Dark slate
    COLOR_SIDEWALK = "#e2e8f0"     # Light stone
    COLOR_CAR = "#2563eb"          # Vibrant blue
    COLOR_LIDAR = "#60a5fa"        # Cyan-blue ring
    COLOR_NOMINAL = "#0f172a"      # Dark navy
    PED_COLORS = ["#ef4444", "#f97316", "#8b5cf6", "#10b981", "#ec4899"]

    fig, axes = plt.subplots(num_snaps, 1, figsize=(14, 3.6 * num_snaps), dpi=130)
    if num_snaps == 1:
        axes = [axes]

    fig.suptitle(
        f"Harvested Pedestrian Threat Encounter Snapshots (Total: {num_snaps})",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    for idx, (snap, ax) in enumerate(zip(snapshots, axes)):
        v_x = snap["vehicle_x"]
        v_y = snap["vehicle_y"]
        D = snap["detected_count"]
        sim_time = snap["sim_time"]

        # Local window centered around vehicle: [vx - 22m, vx + 28m]
        x_min_win = max(0.0, v_x - 20.0)
        x_max_win = min(120.0, v_x + 28.0)

        # 1. Background Sidewalks
        ax.set_facecolor(COLOR_SIDEWALK)

        # 2. Road Asphalt Surface
        road_rect = Rectangle(
            (x_min_win - 5, ROAD_BOTTOM),
            (x_max_win - x_min_win + 10),
            ROAD_WIDTH,
            facecolor=COLOR_ASPHALT,
            edgecolor="#64748b",
            linewidth=1.5,
            zorder=1
        )
        ax.add_patch(road_rect)

        # 3. Dashed Centerline
        centerline_xs = np.arange(x_min_win - 5, x_max_win + 10, 5.0)
        for cx in centerline_xs:
            ax.plot([cx, cx + 3.0], [ROAD_Y, ROAD_Y], color="#f8fafc", linestyle="-", linewidth=1.5, zorder=2)

        # 4. LiDAR Detection Range Circle
        lidar_circle = Circle(
            (v_x, v_y),
            LIDAR_R,
            facecolor="#93c5fd",
            alpha=0.15,
            edgecolor=COLOR_LIDAR,
            linestyle="--",
            linewidth=1.8,
            zorder=3
        )
        ax.add_patch(lidar_circle)

        # 5. Ego Vehicle Body (Rectangle)
        car_rect = Rectangle(
            (v_x - CAR_LENGTH / 2.0, v_y - CAR_WIDTH / 2.0),
            CAR_LENGTH,
            CAR_WIDTH,
            facecolor=COLOR_CAR,
            edgecolor="#1d4ed8",
            linewidth=1.5,
            zorder=6
        )
        ax.add_patch(car_rect)

        # Vehicle Direction Arrow
        ax.arrow(
            v_x + CAR_LENGTH / 2.0, v_y,
            2.5, 0.0,
            head_width=1.0, head_length=0.8,
            fc="#38bdf8", ec="#0284c7",
            linewidth=1.5, zorder=7
        )

        # 6. Plot Pedestrians & Trajectory Scenario Fans
        for d in range(D):
            p0 = snap["ped_positions"][d]
            v0 = snap["ped_velocities"][d]
            nom = snap["nominal_trajectories"][d]  # (N, 2)
            scens = snap["scenarios"][d]           # (S, N, 2)
            color = PED_COLORS[d % len(PED_COLORS)]

            # A. Draw all 100 future scenario paths (semi-transparent uncertainty fan)
            for s_idx in range(len(scens)):
                ax.plot(
                    scens[s_idx, :, 0], scens[s_idx, :, 1],
                    color=color, alpha=0.08, linewidth=1.0,
                    zorder=4
                )

            # B. Draw Nominal Constant-Velocity Trajectory (Bold dashed)
            ax.plot(
                nom[:, 0], nom[:, 1],
                color=COLOR_NOMINAL, linestyle="--", linewidth=2.0,
                label="Nominal (Physics CV)" if (d == 0 and idx == 0) else None,
                zorder=5
            )

            # C. Pedestrian Current Position Marker
            ax.scatter(
                p0[0], p0[1],
                s=90, color=color, edgecolors="#111827",
                linewidth=1.5, zorder=8
            )

            # Direction arrow for pedestrian velocity
            speed_mag = np.hypot(v0[0], v0[1])
            if speed_mag > 0.1:
                ax.arrow(
                    p0[0], p0[1],
                    v0[0] * 0.8, v0[1] * 0.8,
                    head_width=0.7, head_length=0.5,
                    fc=color, ec="#111827",
                    linewidth=1.2, zorder=9
                )

            # Annotation text for pedestrian
            rel_gap_x = p0[0] - v_x
            ax.text(
                p0[0], p0[1] + 1.2,
                f"Ped {d+1}\n(Δx={rel_gap_x:+.1f}m)",
                fontsize=9, fontweight="bold", color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.85, edgecolor="#cbd5e1"),
                ha="center", zorder=10
            )

        # 7. Axes Labels & Limits
        ax.set_xlim(x_min_win, x_max_win)
        ax.set_ylim(8.0, 32.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position (meters)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y Position (meters)", fontsize=10, fontweight="bold")

        # Snapshot Title & Meta
        ax.set_title(
            f"Snapshot #{snap['snapshot_id']}  |  Sim Time: {sim_time:.1f} s (Frame {snap['frame']})  |  "
            f"Vehicle: x = {v_x:.1f} m, v = {snap['vehicle_v']:.1f} m/s  |  "
            f"Detected Threats: {D} pedestrian(s)",
            fontsize=11, fontweight="bold", loc="left", pad=8
        )

        ax.grid(True, linestyle=":", alpha=0.4, color="#94a3b8")

    # Global Legend at Top
    custom_legend = [
        Line2D([0], [0], color=COLOR_CAR, lw=6, label="Ego Vehicle (8.0 m/s Cruise)"),
        Line2D([0], [0], color=COLOR_LIDAR, lw=2, linestyle="--", label="LiDAR Range (R = 12m)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PED_COLORS[0], markersize=9, markeredgecolor="#111", label="Detected Pedestrian"),
        Line2D([0], [0], color=COLOR_NOMINAL, lw=2, linestyle="--", label="Nominal Rollout (CV Physics)"),
        Line2D([0], [0], color=PED_COLORS[0], lw=2, alpha=0.5, label="100 Anchored Scenarios (Uncertainty Fan)"),
    ]
    fig.legend(
        handles=custom_legend,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.995),
        ncol=5,
        fontsize=9,
        framealpha=0.9
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_file = current_dir / output_image
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Visualization saved successfully to: {out_file.resolve()}")

    if show_plot:
        plt.show()

    plt.close()
    return out_file


if __name__ == "__main__":
    visualize_snapshots(show_plot=False)
