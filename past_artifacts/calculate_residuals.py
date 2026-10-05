import os
import sys
from pathlib import Path
import numpy as np

# Ensure environment_setup is importable for DT
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / "environment_setup"))

from units import DT


def calculate_residuals(
    dataset_file="pedestrian_dataset.npy",
    residuals_out="residuals.npy",
    stats_out="residual_stats.npz",
):
    """Calculates deviations (residuals) of true trajectories from Constant Velocity nominal predictions.

    Parameters
    ----------
    dataset_file : str
        Input dataset .npy file path.
    residuals_out : str
        Output file for full residuals array (S, T, 4).
    stats_out : str
        Output file for empirical mean and covariance per timestep.

    Returns
    -------
    W : ndarray, shape (S, T, 4)
        Residuals array (actual - nominal).
    mean_W : ndarray, shape (T, 4)
        Empirical mean residual vector per timestep.
    cov_W : ndarray, shape (T, 4, 4)
        Covariance matrix per timestep.
    """
    # Resolve file paths
    in_path = current_dir / dataset_file if not os.path.isabs(dataset_file) else Path(dataset_file)
    res_path = current_dir / residuals_out if not os.path.isabs(residuals_out) else Path(residuals_out)
    stat_path = current_dir / stats_out if not os.path.isabs(stats_out) else Path(stats_out)

    print(f"Loading dataset from '{in_path}'...")
    Z_true = np.load(in_path)  # Shape: (S, T, 4)
    S, K, num_features = Z_true.shape
    horizon = K - 1
    duration_s = horizon * DT

    print(f"==================================================")
    print(f"Calculating Behavioral Residuals")
    print(f"Samples (S):          {S}")
    print(f"Total States (T):     {K} (t=0 to t={horizon})")
    print(f"Horizon (N):          {horizon} steps ({duration_s:.1f} s at DT={DT}s)")
    print(f"Features:             [x, y, vx, vy]")
    print(f"==================================================")

    # 1. Extract anchor state at t=0 for all S samples: (S, 4)
    Z_0 = Z_true[:, 0, :]

    # 2. Build Nominal Constant Velocity Prediction: (S, K, 4)
    print("Generating nominal predictions (Constant Velocity Model)...")
    Z_hat = np.zeros_like(Z_true)

    for t in range(K):
        elapsed_time = t * DT
        # x_hat = x0 + vx0 * t
        Z_hat[:, t, 0] = Z_0[:, 0] + Z_0[:, 2] * elapsed_time
        # y_hat = y0 + vy0 * t
        Z_hat[:, t, 1] = Z_0[:, 1] + Z_0[:, 3] * elapsed_time
        # Velocity is assumed constant in the nominal model
        Z_hat[:, t, 2] = Z_0[:, 2]
        Z_hat[:, t, 3] = Z_0[:, 3]

    # 3. Calculate Residuals: W = True - Nominal
    print("Calculating residuals (W = True - Nominal)...")
    W = Z_true - Z_hat  # Shape: (S, K, 4)

    # 4. Extract Empirical Mean and Covariance per timestep
    print("Computing residual statistics per timestep...")
    mean_W = np.zeros((K, num_features))
    cov_W = np.zeros((K, num_features, num_features))

    for t in range(K):
        mean_W[t] = np.mean(W[:, t, :], axis=0)
        # Covariance across the S samples for timestep t
        cov_W[t] = np.cov(W[:, t, :].T)

    # 5. Save results
    np.save(res_path, W)
    np.savez(stat_path, mean=mean_W, cov=cov_W)

    print(f"Done! Saved '{res_path.name}' and '{stat_path.name}'.")
    print(f"Max error at t={horizon} (X): {np.max(np.abs(W[:, -1, 0])):.2f} meters")
    print(f"Max error at t={horizon} (Y): {np.max(np.abs(W[:, -1, 1])):.2f} meters")

    return W, mean_W, cov_W


if __name__ == "__main__":
    calculate_residuals()
