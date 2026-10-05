import sys
from pathlib import Path
import numpy as np

# Ensure local directory is importable
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from scenario_generation import generate_scenarios


def predict_detected_pedestrians(
    vehicle_center,
    detected_indices,
    pedestrian_positions,
    pedestrian_velocities,
    residuals,
    horizon=20,
):
    """Generate future trajectory scenarios for all detected pedestrians.

    For each pedestrian within the LiDAR radius, this function uses
    scenario_generation to produce S candidate future trajectories anchored
    to that pedestrian's live state.

    Parameters
    ----------
    vehicle_center : array-like, shape (2,)
        Current vehicle position [x, y].
    detected_indices : array-like, shape (D,)
        Indices of pedestrians detected by LiDAR.
    pedestrian_positions : ndarray, shape (N_ped, 2)
        Current [x, y] positions of ALL pedestrians in environment.
    pedestrian_velocities : ndarray, shape (N_ped, 2)
        Current [vx, vy] velocities of ALL pedestrians.
    residuals : ndarray, shape (S, horizon, 2)
        Historical position residuals (position-only, t=1..horizon).
    horizon : int
        Number of future steps (default: 20 steps = 4.0s).

    Returns
    -------
    result : dict with keys:
        "detected_indices" : ndarray, shape (D,)
            Which pedestrians were processed.
        "scenarios" : ndarray, shape (D, S, horizon, 2)
            For each detected pedestrian, S candidate future
            [x, y] trajectories over the horizon.
        "num_detected" : int
            Number of detected pedestrians (D).
    """
    detected_indices = np.asarray(detected_indices)
    D = len(detected_indices)
    S = residuals.shape[0]

    if D == 0:
        return {
            "detected_indices": detected_indices,
            "scenarios": np.empty((0, S, horizon, 2)),
            "num_detected": 0,
        }

    # For each detected pedestrian, generate S scenarios
    all_scenarios = np.zeros((D, S, horizon, 2))

    for i, ped_idx in enumerate(detected_indices):
        p0 = pedestrian_positions[ped_idx]
        v0 = pedestrian_velocities[ped_idx]

        # generate_scenarios returns (S, horizon, 2)
        all_scenarios[i] = generate_scenarios(p0, v0, residuals, horizon=horizon)

    return {
        "detected_indices": detected_indices,
        "scenarios": all_scenarios,
        "num_detected": D,
    }
