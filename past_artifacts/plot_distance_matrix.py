import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
matrix_path = current_dir / "distance_matrix.npy"

if not matrix_path.exists():
    print(f"File not found: {matrix_path}")
    print("Please run compute_distance_matrix.py first!")
    exit(1)

# Load the distance matrix
D = np.load(matrix_path)
S = D.shape[0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Heatmap of the full distance matrix
ax = axes[0]
im = ax.imshow(D, cmap="viridis", aspect="auto")
ax.set_title(f"Pairwise Distance Matrix ({S}x{S})")
ax.set_xlabel("Scenario j")
ax.set_ylabel("Scenario i")
plt.colorbar(im, ax=ax, label="Normalized Distance")

# 2. Distribution of all pairwise distances
ax = axes[1]
upper = D[np.triu_indices(S, k=1)]
ax.hist(upper, bins=min(60, len(upper) // 10 + 10), color="steelblue", edgecolor="black", linewidth=0.3)
ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Mean = 1.0")
ax.set_title("Distribution of Pairwise Distances")
ax.set_xlabel("Normalized Mahalanobis Distance")
ax.set_ylabel("Count")
ax.legend()

# 3. Average distance of each scenario to all others
ax = axes[2]
avg_dist = D.mean(axis=1)  # (S,)
ax.bar(range(S), avg_dist, color="coral", width=1.0)
ax.axhline(1.0, color="red", linestyle="--", linewidth=2, label="Global Mean = 1.0")
ax.set_title("Per-Scenario Average Distance to All Others")
ax.set_xlabel("Scenario Index")
ax.set_ylabel("Average Distance")
ax.legend()

fig.suptitle(
    f"DR-MPC Ambiguity Metric: Normalized Mahalanobis Distance Matrix\n"
    f"Ledoit-Wolf Shrinkage | {S} Scenarios | Mean Distance = 1.0",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
