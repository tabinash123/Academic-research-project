import sys
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
from threat_assessment import predict_detected_pedestrians

# ======================================================
# Setup
# ======================================================
env = Environment(num_pedestrians=5)
artifacts_dir = parent_dir / "past_artifacts"
residuals = np.load(artifacts_dir / "residuals.npy")[:, 1:21, :2]  # (100, 20, 2)
S, N, _ = residuals.shape

NUM_CYCLES = 5
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

PREDICT_EVERY = 2  # Update predictions every 2 frames (0.4s)

# ======================================================
# Build the figure
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

title = ax.set_title("", fontsize=11)

# Prediction lines collection
prediction_lines = LineCollection([], colors="orange", alpha=0.20, linewidths=1.0, zorder=2)
ax.add_collection(prediction_lines)

# ======================================================
# Animation Loop
# ======================================================
def update(frame):
    env.step()
    v = env.vehicle

    # Update car and lidar
    car_patch.set_xy((v.x, v.y - v.width / 2))
    lidar_patch.set_center(v.center)

    # Update pedestrian positions
    ped_scatter.set_offsets(env.pedestrians.positions)

    # Color detected pedestrians
    detections = env.detections
    if detections is not None:
        colors = np.where(detections["mask"], "lime", "red")
        detected_count = len(detections["indices"])
    else:
        colors = ["red"] * env.pedestrians.n
        detected_count = 0
    ped_scatter.set_color(colors)

    # Update predictions every PREDICT_EVERY frames
    if frame % PREDICT_EVERY == 0 and detections is not None and detected_count > 0:
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

        segments = []
        scenarios = result["scenarios"]  # (D, 100, 20, 2)
        D = result["num_detected"]

        for d in range(D):
            ped_pos = env.pedestrians.positions[detections["indices"][d]]
            for s in range(0, S, max(1, S // 20)):
                traj = scenarios[d, s]  # (20, 2)
                full_path = np.vstack([ped_pos[None, :], traj])
                segments.append(full_path)

        prediction_lines.set_segments(segments)

    elif frame % PREDICT_EVERY == 0 and detected_count == 0:
        prediction_lines.set_segments([])

    cycle = frame // frames_per_cycle + 1
    title.set_text(
        f"Live DR-MPC Scenario Prediction | Cycle: {cycle}/{NUM_CYCLES} | "
        f"Detected Pedestrians: {detected_count} | Frame: {frame}/{total_frames}"
    )

    return [car_patch, lidar_patch, ped_scatter, title, prediction_lines]


if __name__ == "__main__":
    ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)
    plt.show()
