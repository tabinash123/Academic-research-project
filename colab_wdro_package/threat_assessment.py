import numpy as np
from scenario_generation import generate_scenarios


def predict_detected_pedestrians(
    vehicle_center,
    detected_indices,
    pedestrian_positions,
    pedestrian_velocities,
    residuals,
    horizon=20,
    dt=0.2
):
    """Generate future trajectory scenarios for all detected pedestrians within LiDAR radius.

    Parameters
    ----------
    vehicle_center : array-like, shape (2,)
        Current vehicle position [x, y].
    detected_indices : array-like, shape (D,)
        Indices of pedestrians detected by LiDAR.
    pedestrian_positions : ndarray, shape (N_ped, 2)
        Current [x, y] positions of all pedestrians.
    pedestrian_velocities : ndarray, shape (N_ped, 2)
        Current [vx, vy] velocities of all pedestrians.
    residuals : ndarray, shape (S, horizon, 2)
        Historical position residuals.
    horizon : int
        Lookahead horizon steps.

    Returns
    -------
    result : dict with keys:
        'detected_indices': ndarray, shape (D,)
        'scenarios': ndarray, shape (D, S, horizon, 2)
        'num_detected': int
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

    all_scenarios = np.zeros((D, S, horizon, 2))
    for i, ped_idx in enumerate(detected_indices):
        p0 = pedestrian_positions[ped_idx]
        v0 = pedestrian_velocities[ped_idx]
        all_scenarios[i] = generate_scenarios(p0, v0, residuals, dt=dt, horizon=horizon)

    return {
        "detected_indices": detected_indices,
        "scenarios": all_scenarios,
        "num_detected": D,
    }
