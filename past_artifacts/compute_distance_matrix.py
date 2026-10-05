import os
import sys
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import cdist

current_dir = Path(__file__).resolve().parent


def compute_distance_matrix(residuals, out_file="distance_matrix.npy"):
    """Compute the pairwise Mahalanobis distance matrix from historical residuals.

    One-time offline computation. The resulting (S, S) matrix is normalized
    so that the mean pairwise distance equals 1.0.

    Parameters
    ----------
    residuals : ndarray, shape (S, N, 2)
        Position-only residuals for future steps (t=1..N), sliced as [:, 1:N+1, :2].
    out_file : str
        Path or filename to save the resulting (S, S) distance matrix.

    Returns
    -------
    D : ndarray, shape (S, S)
        Normalized pairwise Mahalanobis distance matrix.
    """
    S, N, d = residuals.shape
    feature_dim = N * d

    print(f"==================================================")
    print(f"Computing Pairwise Mahalanobis Distance Matrix")
    print(f"Scenarios (S):        {S}")
    print(f"Horizon (N):          {N} steps")
    print(f"Feature Dimensions:   {feature_dim} ({N} steps x {d} [x, y])")
    print(f"==================================================")

    # Step 1: Flatten (S, N, 2) -> (S, N*2)
    X = residuals.reshape(S, -1)
    print(f"Flattened residuals matrix shape: {X.shape}")

    # Step 2: Fit Ledoit-Wolf shrinkage covariance estimator
    print("Fitting Ledoit-Wolf shrinkage covariance estimator...")
    lw = LedoitWolf().fit(X)
    Sigma = lw.covariance_  # (N*2, N*2)
    shrinkage = lw.shrinkage_
    print(f"Shrinkage coefficient: {shrinkage:.4f}")

    # Step 3: Invert covariance matrix
    Sigma_inv = np.linalg.inv(Sigma)
    print("Covariance inverted successfully.")

    # Step 4: Compute full pairwise Mahalanobis distance matrix
    print(f"Computing {S}x{S} pairwise Mahalanobis distances...")
    D = cdist(X, X, metric="mahalanobis", VI=Sigma_inv)

    # Step 5: Normalize so average pairwise distance equals 1.0
    mean_dist = D.mean()
    D = D / mean_dist
    print(f"Normalized by mean pairwise distance ({mean_dist:.4f}).")

    # Step 6: Save
    save_path = current_dir / out_file if not os.path.isabs(out_file) else Path(out_file)
    np.save(save_path, D)
    print(f"Saved distance matrix of shape {D.shape} to '{save_path.name}'.")

    return D


if __name__ == "__main__":
    res_path = current_dir / "residuals.npy"
    if not res_path.exists():
        print(f"Error: '{res_path}' not found! Run calculate_residuals.py first.")
        sys.exit(1)

    # Load full residuals (S, K, 4)
    full_residuals = np.load(res_path)
    S, K, _ = full_residuals.shape
    horizon = K - 1  # 20

    # Slice to future position-only residuals (t=1..horizon)
    residuals_pos = full_residuals[:, 1 : horizon + 1, :2]  # (S, 20, 2)

    D = compute_distance_matrix(residuals_pos)

    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    print(f"  Shape:               {D.shape}")
    print(f"  Diagonal max (0.0):  {D.diagonal().max():.6f}")
    print(f"  Is Symmetric:        {np.allclose(D, D.T)}")
    print(f"  Mean distance (1.0): {D.mean():.4f}")
    print(f"  Min non-zero:        {D[D > 0].min():.4f}")
    print(f"  Max distance:        {D.max():.4f}")
