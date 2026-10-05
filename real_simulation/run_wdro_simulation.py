"""
Full Closed-Loop Simulation with WDRO-MPC controller driving the vehicle.
Perceives pedestrians within LiDAR radius, generates anchored scenario trajectories,
solves the Wasserstein DR-MPC problem using Clarabel, and applies real-time acceleration.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# Set up module search paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir / "environment_setup"))
sys.path.insert(0, str(parent_dir / "past_artifacts"))

from environment import Environment
from units import sim_speed_to_real, DT
from threat_assessment import predict_detected_pedestrians
from wdro_mpc import solve_wdro_mpc

# ======================================================
# Setup & Load Precomputed Data
# ======================================================
env = Environment(num_pedestrians=5)  # 5 test pedestrians for live evaluation

artifacts_dir = parent_dir / "past_artifacts"
res_file = artifacts_dir / "residuals.npy"
dist_file = artifacts_dir / "distance_matrix.npy"

if not res_file.exists() or not dist_file.exists():
    raise FileNotFoundError("Missing precomputed offline data in past_artifacts/ directory!")

# Load and slice future position residuals: (100, 20, 2)
residuals = np.load(res_file)[:, 1:21, :2]
dist_matrix = np.load(dist_file)  # (100, 100)

S, N, _ = residuals.shape  # S=100, N=20

NUM_CYCLES = 5
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

CONTROL_EVERY = 2  # Solve MPC every 2 frames (0.4s) for high efficiency

# Store the last MPC acceleration to apply between solves
current_accel = 0.0

# ======================================================
# Build the Figure & Visual Artists
# ======================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, env.world_width)
ax.set_ylim(0, env.world_height)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor("#d8f0d0")

# Static: road
road = Rectangle(
    (0, env.road_y - env.road_width / 2),
    env.world_width, env.road_width, color="gray"
)
ax.add_patch(road)
ax.plot([0, env.world_width], [env.road_y, env.road_y], "--", color="white", linewidth=2)

# Dynamic elements
v = env.vehicle

car_patch = Rectangle(
    (v.x, v.y - v.width / 2), v.length, v.width, color="blue", zorder=4
)
ax.add_patch(car_patch)

lidar_patch = Circle(
    v.center, env.lidar.radius,
    fill=False, linestyle="--", edgecolor="cyan", linewidth=2, zorder=3
)
ax.add_patch(lidar_patch)

ped_scatter = ax.scatter(
    env.pedestrians.x, env.pedestrians.y,
    s=200, c=["red"] * env.pedestrians.n,
    edgecolors="black", linewidths=0.5, zorder=5
)

# Prediction lines (orange fan)
prediction_lines = LineCollection([], colors="orange", alpha=0.18, linewidths=1.0, zorder=2)
ax.add_collection(prediction_lines)

title = ax.set_title("", fontsize=10)

# ======================================================
# Animation Loop
# ======================================================
def update(frame):
    global current_accel

    v = env.vehicle
    detections = env.detections

    # Run MPC solver at control intervals when pedestrians are detected
    if frame % CONTROL_EVERY == 0 and detections is not None and len(detections["indices"]) > 0:
        speed_real = env.pedestrians.speed_real
        heading = env.pedestrians.heading
        velocities = np.column_stack([
            speed_real * np.cos(heading),
            speed_real * np.sin(heading)
        ])

        result = predict_detected_pedestrians(
            vehicle_center=v.center,
            detected_indices=detections["indices"],
            pedestrian_positions=env.pedestrians.positions,
            pedestrian_velocities=velocities,
            residuals=residuals,
            horizon=N
        )

        scenarios_all = result["scenarios"]  # (D, 100, 20, 2)
        scenarios_x_all = scenarios_all[:, :, :, 0]  # (D, 100, 20)

        a_opt, info = solve_wdro_mpc(
            x0=v.center[0],
            v0=sim_speed_to_real(v.speed_sim),
            scenarios_x_all=scenarios_x_all,
            dist_matrix=dist_matrix,
            dt=DT,
            N=N
        )

        current_accel = a_opt if a_opt is not None else 0.0

        # Update prediction fan lines
        segments = []
        for d in range(result["num_detected"]):
            ped_pos = env.pedestrians.positions[detections["indices"][d]]
            for s in range(0, S, max(1, S // 20)):  # Draw 20 scenario lines per pedestrian
                traj = scenarios_all[d, s]  # (20, 2)
                full_path = np.vstack([ped_pos[None, :], traj])
                segments.append(full_path)
        prediction_lines.set_segments(segments)

    elif frame % CONTROL_EVERY == 0:
        # Cruising with no pedestrians detected
        current_accel = 0.0
        prediction_lines.set_segments([])

    # Step the environment with controller's acceleration
    env.step(accel_real=current_accel)

    # Update visual patches
    v = env.vehicle
    car_patch.set_xy((v.x, v.y - v.width / 2))
    lidar_patch.set_center(v.center)
    ped_scatter.set_offsets(env.pedestrians.positions)

    detections = env.detections
    if detections is not None:
        colors = np.where(detections["mask"], "lime", "red")
        detected_count = len(detections["indices"])
    else:
        colors = ["red"] * env.pedestrians.n
        detected_count = 0
    ped_scatter.set_color(colors)

    cycle = frame // frames_per_cycle + 1
    speed_ms = sim_speed_to_real(v.speed_sim)

    # Color car based on controller decision
    if current_accel < -0.5:
        car_color = "red"       # braking
    elif current_accel > 0.5:
        car_color = "green"     # accelerating
    else:
        car_color = "blue"      # cruising
    car_patch.set_facecolor(car_color)

    title.set_text(
        f"WDRO-MPC Controller | Cycle: {cycle}/{NUM_CYCLES} | "
        f"Speed: {speed_ms:.1f} m/s | Accel: {current_accel:+.2f} m/s² | "
        f"Detected: {detected_count} | Frame: {frame}/{total_frames}"
    )

    return [car_patch, lidar_patch, ped_scatter, title, prediction_lines]


if __name__ == "__main__":
    print(f"Starting WDRO-MPC simulation ({total_frames} frames across {NUM_CYCLES} cycles)...")
    ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)
    plt.show()
