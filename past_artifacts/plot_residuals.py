import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

current_dir = Path(__file__).resolve().parent


def plot_residuals(res_file="residuals.npy", stats_file="residual_stats.npz"):
    r_path = current_dir / res_file if not os.path.isabs(res_file) else Path(res_file)
    s_path = current_dir / stats_file if not os.path.isabs(stats_file) else Path(stats_file)

    print(f"Loading {r_path.name} and {s_path.name}...")
    W = np.load(r_path)  # Shape (S, T, 4)
    stats = np.load(s_path)
    mean_W = stats["mean"]  # Shape (T, 4)
    cov_W = stats["cov"]  # Shape (T, 4, 4)

    S, K, features = W.shape
    DT = 0.2
    duration_s = (K - 1) * DT

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    print(f"Plotting all {S} residual trajectories...")
    for idx in range(S):
        plt.plot(W[idx, :, 0], W[idx, :, 1], "k-", alpha=0.15, linewidth=1.0)

    # Start point t=0
    plt.plot(0, 0, "go", markersize=8, label="t=0 (Initial State, Error=0)")

    def draw_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
        cov_xy = cov[0:2, 0:2]
        mean_xy = mean[0:2]
        vals, vecs = np.linalg.eigh(cov_xy)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-6))
        ellipse = Ellipse(xy=mean_xy, width=width, height=height, angle=theta, **kwargs)
        ax.add_patch(ellipse)

    print("Drawing uncertainty bounds (covariance ellipses)...")
    # Draw 2-sigma (95% confidence) bounds for 1.0s (t=5), 2.0s (t=10), 4.0s (t=20)
    step_1s = min(5, K - 1)
    step_2s = min(10, K - 1)
    step_4s = min(20, K - 1)

    timesteps = {
        step_1s: ("blue", f"{step_1s * DT:.1f}s"),
        step_2s: ("orange", f"{step_2s * DT:.1f}s"),
        step_4s: ("red", f"{step_4s * DT:.1f}s"),
    }

    for t_step, (color, time_str) in timesteps.items():
        draw_ellipse(
            ax,
            mean_W[t_step],
            cov_W[t_step],
            n_std=2.0,
            edgecolor=color,
            facecolor="none",
            linewidth=2.5,
            label=f"2σ Bound at t={time_str}",
        )

    plt.title(
        f"DR-MPC Uncertainty: Behavioral Residuals over {duration_s:.1f}s (S={S}, DT={DT}s)\n"
        f"Deviation of true pedestrian motion from Constant Velocity nominal model"
    )
    plt.xlabel("Error in X Position (meters)")
    plt.ylabel("Error in Y Position (meters)")

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_residuals()
