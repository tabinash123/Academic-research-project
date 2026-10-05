import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

current_dir = Path(__file__).resolve().parent


def plot_trajectories(npy_file="pedestrian_dataset.npy"):
    file_path = current_dir / npy_file if not os.path.isabs(npy_file) else Path(npy_file)
    print(f"Loading {file_path}...")
    dataset = np.load(file_path)
    M, K, features = dataset.shape
    DT = 0.2
    duration_s = (K - 1) * DT

    plt.figure(figsize=(12, 6))

    # Road boundaries
    ROAD_Y = 20
    ROAD_WIDTH = 12
    MARGIN = 8

    plt.axhline(ROAD_Y - ROAD_WIDTH / 2, color="gray", linestyle="--", linewidth=2, label="Road Edge")
    plt.axhline(ROAD_Y + ROAD_WIDTH / 2, color="gray", linestyle="--", linewidth=2)
    plt.axhline(ROAD_Y, color="lightgray", linestyle=":")

    plt.axhline(ROAD_Y - ROAD_WIDTH / 2 - MARGIN, color="brown", linestyle=":", alpha=0.5, label="Sidewalk Limit")
    plt.axhline(ROAD_Y + ROAD_WIDTH / 2 + MARGIN, color="brown", linestyle=":", alpha=0.5)

    print(f"Plotting all {M} trajectories ({duration_s:.1f}s horizon)...")
    for idx in range(M):
        traj = dataset[idx]  # Shape: (K, 4) -> [x, y, vx, vy]
        x = traj[:, 0]
        y = traj[:, 1]

        plt.plot(x, y, alpha=0.25, linewidth=1.2)
        plt.plot(x[0], y[0], "go", markersize=3, alpha=0.5)
        plt.plot(x[-1], y[-1], "ro", markersize=3, alpha=0.5)

    plt.title(f"All {M} Historical Pedestrian Trajectories (N={K-1} steps, {duration_s:.1f}s lookahead, DT={DT}s)")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")

    custom_lines = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2),
        Line2D([0], [0], color="brown", linestyle=":", alpha=0.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="g", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=8),
    ]
    plt.legend(custom_lines, ["Road Edge", "Sidewalk Limit", "Start (t=0)", f"End (t={K-1})"], loc="upper right")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_trajectories()
