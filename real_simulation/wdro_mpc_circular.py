"""Circular Wasserstein Distributionally Robust Model Predictive Control (WDRO-MPC) Solver.

Implements the circular formulation where the encounter distance d_k is evaluated
against the actual planned vehicle trajectory x_k (rather than a frozen constant-speed baseline).

The circular dependency:
  Acceleration a -> Vehicle Position x -> Distance d -> Target Speed v_desired -> Loss J -> Acceleration a
is solved via Iterative Sequential Convex Programming (SCP) until the planned trajectory
and the distance calculations converge to self-consistency.

This version is fully vectorized for better performance.
"""

import time
import numpy as np
import cvxpy as cp


def solve_wdro_mpc_circular(
    x0,
    v0,
    scenarios,
    dist_matrix,
    v_des=8.0,
    v_max=15.0,
    v_min=0.0,
    c=5.0,
    w=0.5,
    epsilon=0.8,
    a_min=-8.0,
    a_max=4.0,
    road_y=20.0,
    dt=0.2,
    N=20,
    max_iter=3,
    conv_tol=0.05,
    solver=cp.CLARABEL,
):
    """Solves the circular WDRO-MPC optimization problem iteratively (vectorized).

    Parameters
    ----------
    x0 : float
        Current vehicle position (meters).
    v0 : float
        Current vehicle speed (m/s).
    scenarios : ndarray
        Pedestrian scenarios of shape (D, S, N, 2) or (D, S, N).
    dist_matrix : ndarray, shape (S, S)
        Ground transport cost matrix between scenarios.
    v_des : float, optional
        Target cruising speed (default: 8.0 m/s).
    v_max : float, optional
        Maximum speed limit (default: 15.0 m/s).
    v_min : float, optional
        Minimum speed limit (default: 0.0 m/s).
    c : float, optional
        Distance scaling threshold (default: 5.0 m).
    w : float, optional
        Acceleration effort penalty weight (default: 0.5).
    epsilon : float, optional
        Wasserstein ambiguity ball radius (default: 0.1).
    a_min, a_max : float, optional
        Actuator acceleration bounds (default: -5.0, +3.0 m/s^2).
    road_y : float, optional
        Road centerline lateral coordinate (default: 20.0 m).
    dt : float, optional
        Sampling period (default: 0.2 s).
    N : int, optional
        Prediction horizon (default: 20 steps).
    max_iter : int, optional
        Maximum circular convergence iterations (default: 3).
    conv_tol : float, optional
        Position convergence threshold in meters (default: 0.05 m = 5 cm).
    solver : cp.Solver, optional
        Convex solver (default: cp.CLARABEL).

    Returns
    -------
    a_opt : float or None
        Optimal initial acceleration command a_0*.
    info : dict
        Detailed convergence and telemetry dictionary.
    """
    t_start = time.perf_counter()

    # 1. Normalize scenario input shape to (D, S, N, features)
    scenarios_arr = np.asarray(scenarios, dtype=float)
    if scenarios_arr.ndim == 2:
        scenarios_arr = scenarios_arr[None, :, :]
    elif scenarios_arr.ndim == 3 and scenarios_arr.shape[-1] == 2:
        scenarios_arr = scenarios_arr[None, :, :, :]

    D = scenarios_arr.shape[0] if scenarios_arr.size > 0 else 0
    S = dist_matrix.shape[0]

    # 2. Warm-start planned vehicle trajectory: initial guess assuming constant velocity
    t_steps = np.arange(1, N + 1) * dt
    planned_x = x0 + v0 * t_steps  # shape (N,)

    final_info = None
    final_a_opt = None
    iter_count = 0

    # 3. Iterative Sequential Convex Programming (SCP) Loop
    for iteration in range(max_iter):
        iter_count += 1

        # ---------- Distance & desired speed (vectorized) ----------
        if D > 0:
            if scenarios_arr.ndim == 4 and scenarios_arr.shape[-1] == 2:
                ped_x = scenarios_arr[:, :, :, 0]          # (D, S, N)
                ped_y = scenarios_arr[:, :, :, 1]          # (D, S, N)
                dx = ped_x - planned_x[None, None, :]     # broadcasting
                dy = ped_y - road_y
                dist_all = np.sqrt(dx**2 + dy**2)         # (D, S, N)
                d_k = np.min(dist_all, axis=0)            # closest hazard (S, N)
            else:
                ped_x = scenarios_arr if scenarios_arr.ndim == 3 else scenarios_arr[:, :, :, 0]
                dx = ped_x - planned_x[None, None, :]
                d_k = np.min(np.abs(dx), axis=0)          # (S, N)

            # Desired speed from distance (vectorized)
            v_desired = v_des * (d_k / (d_k + c))         # shape (S, N)
        else:
            v_desired = np.full((S, N), v_des, dtype=float)

        # ---------- Build Convex Optimization Program ----------
        a = cp.Variable(N, name="a")
        v = cp.Variable(N + 1, name="v")
        x = cp.Variable(N + 1, name="x")

        constraints = [
            x[0] == x0,
            v[0] == v0,
            x[1:] == x[:-1] + v[:-1] * dt + 0.5 * a * dt**2,
            v[1:] == v[:-1] + a * dt,
            a >= a_min,
            a <= a_max,
            v >= v_min,
            v <= v_max,
        ]

        if D > 0:
            s = cp.Variable(S, name="s")
            lam = cp.Variable(nonneg=True, name="lambda")
            J_loss = cp.Variable(S, name="J_loss")

            # ===== Fully vectorized cost constraints =====
            # tracking_error shape: (S, N)
            tracking_error = v[:N][None, :] - v_desired
            constraints.append(
                J_loss >= cp.sum_squares(tracking_error, axis=1) + w * cp.sum_squares(a)
            )

            # Dual Wasserstein constraints (already vectorized)
            s_col = cp.reshape(s, (S, 1), order="C")
            J_row = cp.reshape(J_loss, (1, S), order="C")
            constraints.append(s_col >= J_row - lam * dist_matrix)

            objective = cp.Minimize((1.0 / S) * cp.sum(s) + epsilon * lam)
        else:
            cost = cp.sum_squares(v[:N] - v_des) + w * cp.sum_squares(a)
            objective = cp.Minimize(cost)
            s = None
            lam = None

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError as e:
            solve_time_ms = (time.perf_counter() - t_start) * 1000.0
            return None, {
                "status": "solver_error",
                "message": str(e),
                "iterations": iter_count,
                "solve_time_ms": solve_time_ms,
            }

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Circular WDRO iteration {iteration+1} status = {problem.status}")
            return None, {"status": problem.status, "iterations": iter_count}

        # Update planned trajectory
        new_planned_x = np.array(x.value[1:])  # steps k = 1 to N
        pos_shift = np.max(np.abs(new_planned_x - planned_x))
        planned_x = new_planned_x

        final_a_opt = float(a.value[0])
        final_info = {
            "status": problem.status,
            "objective": float(problem.value),
            "a_full": np.array(a.value),
            "v_full": np.array(v.value),
            "x_full": np.array(x.value),
            "v_desired": v_desired,
            "lam": float(lam.value) if (D > 0 and lam is not None and lam.value is not None) else None,
            "s_mean": float(np.mean(s.value)) if (D > 0 and s is not None and s.value is not None) else None,
            "iterations": iter_count,
            "pos_shift": float(pos_shift),
        }

        # Check convergence of circular dependency
        if pos_shift < conv_tol:
            break

    solve_time_ms = (time.perf_counter() - t_start) * 1000.0
    final_info["solve_time_ms"] = solve_time_ms

    return final_a_opt, final_info