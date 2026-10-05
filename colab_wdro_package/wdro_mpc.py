"""Wasserstein Distributionally Robust Model Predictive Control (WDRO-MPC) Solver.

Official Mathematical Formulation:
- Distance-Governed Speed Regulation: v_desired_k^(j) = v_des * (d_k^(j) / (d_k^(j) + c))
- Nominal Reference Freezing: Evaluates distance along frozen constant-speed baseline
- Strict Convexity Guarantee: Pure Convex Quadratic Program (QP)
- Dual Wasserstein Ambiguity Constraint: s_i >= J_j - lambda * dist(i, j)

This version is fully vectorized for maximum performance.
"""

import time
import numpy as np
import cvxpy as cp


def solve_wdro_mpc(
    x0,
    v0,
    scenarios,
    dist_matrix,
    v_des=8.0,
    v_max=15.0,
    v_min=0.0,
    c=15.0,
    w=0.2,
    epsilon=0.4,
    a_min=-8.0,
    a_max=3.0,
    road_y=20.0,
    dt=0.2,
    N=20,
    solver=cp.CLARABEL,
):
    """Solves the official WDRO-MPC problem for longitudinal vehicle control (vectorized).

    Parameters
    ----------
    x0 : float
        Current vehicle longitudinal position in meters.
    v0 : float
        Current vehicle longitudinal velocity in m/s.
    scenarios : ndarray
        Predicted pedestrian trajectories. Can be:
        - shape (D, S, N, 2): 2D trajectories [x, y] for D pedestrians
        - shape (D, S, N): 1D longitudinal trajectories [x]
        - shape (S, N, 2) or (S, N): single pedestrian trajectories
    dist_matrix : ndarray, shape (S, S)
        Precomputed pairwise ground transport metric between scenario residuals.
    v_des : float, optional
        Target cruising speed when no pedestrian hazard exists (default: 8.0 m/s).
    v_max : float, optional
        Maximum admissible speed limit (default: 15.0 m/s).
    v_min : float, optional
        Minimum speed bound (default: 0.0 m/s).
    c : float, optional
        Distance scaling threshold parameter in meters (default: 15.0 m).
    w : float, optional
        Weight penalizing acceleration effort / passenger comfort (default: 0.2).
    epsilon : float, optional
        Wasserstein ambiguity ball radius (default: 0.4).
    a_min, a_max : float, optional
        Actuator acceleration and braking bounds in m/s^2.
    road_y : float, optional
        Road centerline lateral coordinate (default: 20.0 m).
    dt : float, optional
        Sampling period in seconds (default: 0.2 s).
    N : int, optional
        Prediction horizon steps (default: 20 steps = 4.0 s).
    solver : cp.Solver, optional
        CVXPY convex solver (default: cp.CLARABEL).

    Returns
    -------
    a_opt : float or None
        Optimal acceleration command for the current step (a[0]).
    info : dict
        Telemetry containing solver status, solve time, optimal trajectories, and dual variables.
    """
    t_start = time.perf_counter()

    # ----------------------------------------------------------
    # 1. Normalize scenario input shape to (D, S, N, features)
    # ----------------------------------------------------------
    scenarios_arr = np.asarray(scenarios, dtype=float)
    if scenarios_arr.ndim == 2:
        scenarios_arr = scenarios_arr[None, :, :]
    elif scenarios_arr.ndim == 3 and scenarios_arr.shape[-1] == 2:
        scenarios_arr = scenarios_arr[None, :, :, :]

    D = scenarios_arr.shape[0] if scenarios_arr.size > 0 else 0
    S = dist_matrix.shape[0]

    # ----------------------------------------------------------
    # 2. Nominal Reference Freezing & Target Velocity Precomputation
    # ----------------------------------------------------------
    t_steps = np.arange(1, N + 1) * dt
    car_x_frozen = x0 + v0 * t_steps          # shape (N,)
    car_y_frozen = road_y

    v_desired = np.full((S, N), v_des, dtype=float)

    if D > 0:
        if scenarios_arr.ndim == 4 and scenarios_arr.shape[-1] == 2:
            # Full 2D Euclidean distance
            ped_x = scenarios_arr[:, :, :, 0]   # (D, S, N)
            ped_y = scenarios_arr[:, :, :, 1]   # (D, S, N)
            dx = ped_x - car_x_frozen[None, None, :]
            dy = ped_y - car_y_frozen
            dist_all = np.sqrt(dx**2 + dy**2)
            d_k = np.min(dist_all, axis=0)      # (S, N)
        else:
            # 1D longitudinal distance
            ped_x = scenarios_arr if scenarios_arr.ndim == 3 else scenarios_arr[:, :, :, 0]
            dx = ped_x - car_x_frozen[None, None, :]
            d_k = np.min(np.abs(dx), axis=0)    # (S, N)

        # Distance-governed desired velocity
        v_desired = v_des * (d_k / (d_k + c))   # shape (S, N)

    # ----------------------------------------------------------
    # 3. Decision Variables
    # ----------------------------------------------------------
    a = cp.Variable(N, name="a")
    v = cp.Variable(N + 1, name="v")

    constraints = [
        v[0] == v0,
        v[1:] == v[:-1] + a * dt,
        a >= a_min,
        a <= a_max,
        v >= v_min,
        v <= v_max,
    ]

    # ----------------------------------------------------------
    # 4. Objective & Dual Wasserstein Constraints (vectorized)
    # ----------------------------------------------------------
    if D > 0:
        s = cp.Variable(S, name="s")
        lam = cp.Variable(nonneg=True, name="lambda")
        J_loss = cp.Variable(S, name="J_loss")

        # ===== Fully vectorized cost constraints =====
        tracking_error = v[:N][None, :] - v_desired          # (S, N)
        constraints.append(
            J_loss >= cp.sum_squares(tracking_error, axis=1) + w * cp.sum_squares(a)
        )

        # Dual Wasserstein epigraph constraints (already vectorized)
        s_col = cp.reshape(s, (S, 1), order="C")
        J_row = cp.reshape(J_loss, (1, S), order="C")
        constraints.append(s_col >= J_row - lam * dist_matrix)

        objective = cp.Minimize((1.0 / S) * cp.sum(s) + epsilon * lam)
    else:
        # Clear road → track desired cruising speed
        cost = cp.sum_squares(v[:N] - v_des) + w * cp.sum_squares(a)
        objective = cp.Minimize(cost)
        s = None
        lam = None

    # ----------------------------------------------------------
    # 5. Solve
    # ----------------------------------------------------------
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=solver, verbose=False)
    except cp.SolverError as e:
        solve_time_ms = (time.perf_counter() - t_start) * 1000.0
        return None, {
            "status": "solver_error",
            "message": str(e),
            "solve_time_ms": solve_time_ms,
            "objective": None,
            "a_full": None,
            "v_full": None,
        }

    solve_time_ms = (time.perf_counter() - t_start) * 1000.0

    info = {
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else None,
        "a_full": np.array(a.value) if a.value is not None else None,
        "v_full": np.array(v.value) if v.value is not None else None,
        "v_desired": v_desired,
        "lam": float(lam.value) if (D > 0 and lam is not None and lam.value is not None) else None,
        "s_mean": float(np.mean(s.value)) if (D > 0 and s is not None and s.value is not None) else None,
        "solve_time_ms": solve_time_ms,
    }

    if problem.status in ["optimal", "optimal_inaccurate"]:
        a_opt = float(a.value[0])
        return a_opt, info
    else:
        print(f"Warning: WDRO-MPC Solver status = {problem.status}")
        return None, info
