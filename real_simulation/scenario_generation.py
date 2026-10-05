import sys
from pathlib import Path
import numpy as np

# Ensure environment_setup is importable for DT
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / "environment_setup"))

try:
    from units import DT
except ImportError:
    DT = 0.2  # Default: 0.2s (5 Hz)


def nominal_rollout(p0, v0, horizon=20, dt=None):
    """Constant Velocity nominal prediction from a single pedestrian state.

    Parameters
    ----------
    p0 : array-like, shape (2,)
        Current pedestrian position [x, y] in meters.
    v0 : array-like, shape (2,)
        Current pedestrian velocity [vx, vy] in m/s.
    horizon : int
        Number of future steps to predict (default 20 = 4.0s).
    dt : float, optional
        Time step in seconds (defaults to units.DT = 0.2s).

    Returns
    -------
    nominal : ndarray, shape (horizon, 2)
        Predicted [x, y] positions at t=1, ..., horizon.
    """
    if dt is None:
        dt = DT

    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    # Time vector for t=1 to t=horizon
    t = np.arange(1, horizon + 1) * dt  # shape (horizon,)

    # Position = p0 + v0 * t  (broadcast: (horizon,1) * (2,) -> (horizon,2))
    nominal = p0 + v0 * t[:, None]

    return nominal


def generate_scenarios(p0, v0, residuals, horizon=20, dt=None):
    """Generate S candidate future trajectories for one pedestrian.

    Takes the current live state from LiDAR, rolls a nominal Constant Velocity
    prediction forward, then overlays every historical residual
    to produce S plausible future paths.

    Parameters
    ----------
    p0 : array-like, shape (2,)
        Current pedestrian position [x, y] in meters (from LiDAR).
    v0 : array-like, shape (2,)
        Current pedestrian velocity [vx, vy] in m/s (from LiDAR).
    residuals : ndarray, shape (S, horizon, 2)
        Historical position residuals (position-only, t=1..horizon).
        Obtained offline via: residuals[:, 1:horizon+1, :2] from residuals.npy.
    horizon : int
        Number of future steps (default 20 = 4.0s).
    dt : float, optional
        Time step in seconds (defaults to units.DT = 0.2s).

    Returns
    -------
    scenarios : ndarray, shape (S, horizon, 2)
        S candidate future [x, y] trajectories for this pedestrian.
    """
    # Step 1: Nominal rollout from live state
    nominal = nominal_rollout(p0, v0, horizon=horizon, dt=dt)  # shape (horizon, 2)

    # Step 2: Anchor every residual to the live nominal
    # nominal is (horizon, 2), residuals is (S, horizon, 2)
    # Broadcasting: (1, horizon, 2) + (S, horizon, 2) -> (S, horizon, 2)
    scenarios = nominal[None, :, :] + residuals

    return scenarios
