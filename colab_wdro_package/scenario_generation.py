import numpy as np


def generate_scenarios(p0, v0, residuals, dt=0.2, horizon=20):
    """Generate future trajectory scenarios for a pedestrian anchored to their live state.

    Parameters
    ----------
    p0 : array-like, shape (2,)
        Pedestrian's current live [x, y] position.
    v0 : array-like, shape (2,)
        Pedestrian's current live [vx, vy] velocity.
    residuals : ndarray, shape (S, horizon, 2)
        Historical position residuals.
    dt : float
        Simulation time step in seconds (default: 0.2s).
    horizon : int
        Lookahead steps (default: 20 steps = 4.0s).

    Returns
    -------
    scenarios : ndarray, shape (S, horizon, 2)
        Anchored future [x, y] trajectories over the horizon.
    """
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    S = residuals.shape[0]

    # Nominal straight-line kinematic future positions: (horizon, 2)
    time_steps = np.arange(1, horizon + 1)[:, None]  # (horizon, 1)
    p_nominal = p0 + v0 * (time_steps * dt)         # (horizon, 2)

    # Anchor historical residuals onto nominal path: (S, horizon, 2)
    scenarios = p_nominal[None, :, :] + residuals[:, :horizon, :]
    return scenarios
