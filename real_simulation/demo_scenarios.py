import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Set up module search paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir / "environment_setup"))
sys.path.insert(0, str(parent_dir / "past_artifacts"))

from environment import Environment
from threat_assessment import predict_detected_pedestrians

# ======================================================
# Setup
# ======================================================
env = Environment(num_pedestrians=5)
artifacts_dir = parent_dir / "past_artifacts"
residuals = np.load(artifacts_dir / "residuals.npy")[:, 1:21, :2]  # (100, 20, 2)
S, N, _ = residuals.shape

NUM_CYCLES = 3
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

SNAPSHOT_INTERVAL = 10  # capture every 10 frames

snapshots = []
print(f"Running {total_frames} frames to capture scenario prediction snapshots...")

for frame in range(total_frames):
    env.step()

    if frame % SNAPSHOT_INTERVAL == 0:
        speed_real = env.pedestrians.speed_real
        heading = env.pedestrians.heading
        velocities = np.column_stack([
            speed_real * np.cos(heading),
            speed_real * np.sin(heading)
        ])

        detections = env.detections
        if detections is not None and len(detections["indices"]) > 0:
            result = predict_detected_pedestrians(
                vehicle_center=env.vehicle.center,
                detected_indices=detections["indices"],
                pedestrian_positions=env.pedestrians.positions,
                pedestrian_velocities=velocities,
                residuals=residuals,
                horizon=N
            )

            snapshots.append({
                "frame": frame,
                "cycle": frame // frames_per_cycle + 1,
                "vehicle_center": env.vehicle.center,
                "lidar_radius": env.lidar.radius,
                "ped_positions": env.pedestrians.positions.copy(),
                "detected_indices": result["detected_indices"].copy(),
                "scenarios": result["scenarios"].copy(),
            })

print(f"Captured {len(snapshots)} snapshots with detected pedestrians.")

if len(snapshots) == 0:
    print("No pedestrians were detected in any snapshot.")
    sys.exit(0)

# Pick up to 6 evenly spaced snapshots to display
num_plots = min(6, len(snapshots))
plot_indices = np.linspace(0, len(snapshots) - 1, num_plots, dtype=int)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax_idx, snap_idx in enumerate(plot_indices):
    ax = axes[ax_idx]
    snap = snapshots[snap_idx]

    vx, vy = snap["vehicle_center"]
    radius = snap["lidar_radius"]

    # Road & Sidewalks
    ax.axhline(env.road_y - env.road_width / 2, color="gray", linestyle="--", linewidth=1.5)
    ax.axhline(env.road_y + env.road_width / 2, color="gray", linestyle="--", linewidth=1.5)

    # LiDAR Circle
    circle = plt.Circle((vx, vy), radius, fill=False, linestyle="--", edgecolor="cyan", linewidth=1.5)
    ax.add_patch(circle)

    # Vehicle
    ax.plot(vx, vy, "s", color="blue", markersize=10, label="Vehicle")

    # All pedestrians
    all_pos = snap["ped_positions"]
    ax.scatter(all_pos[:, 0], all_pos[:, 1], c="grey", s=30, alpha=0.4)

    # Detected pedestrians & their candidate scenario fans
    detected = snap["detected_indices"]
    scenarios = snap["scenarios"]

    for d in range(len(detected)):
        ped_idx = detected[d]
        ped_pos = all_pos[ped_idx]
        ax.plot(ped_pos[0], ped_pos[1], "o", color="red", markersize=8, zorder=5)

        for s in range(0, S, max(1, S // 30)):  # draw up to 30 scenario paths
            traj = scenarios[d, s]
            ax.plot(traj[:, 0], traj[:, 1], color="orange", alpha=0.15, linewidth=1.0)

    ax.set_title(f"Frame {snap['frame']} | {len(detected)} detected", fontsize=10)
    ax.set_xlim(0, env.world_width)
    ax.set_ylim(env.road_y - 15, env.road_y + 15)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

fig.suptitle(
    f"DR-MPC Scenario Generation: Anchored Pedestrian Trajectory Fans\n"
    f"Lookahead N={N} steps (4.0s) | {S} Residual Scenarios",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
