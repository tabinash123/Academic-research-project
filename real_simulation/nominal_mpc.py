"""Deterministic Nominal Model Predictive Control Solver.

Baseline nominal optimization model for benchmarking against WDRO-MPC and SAA-MPC:
- Collapses all S candidate pedestrian prediction scenarios into a single mean trajectory.
- Optimizes a standard single-track quadratic tracking objective against this nominal path.
- Ignores scenario variance and distributional ambiguity entirely.
- Retains identical vehicle kinematics, speed boundaries, and distance-governed target velocity scaling.
- Solved via Sequential Convex Programming (SCP) for self-consistent vehicle-pedestrian distance coupling.
"""

import time
import numpy as np
import cvxpy as cp


def solve_nominal_mpc(
    x0,
    v0,
    scenarios,
    v_des=8.0,
    v_max=15.0,
    v_min=0.0,
    c=20.0,
    w=0.2,
    a_min=-8.0,
    a_max=3.0,
    road_y=20.0,
    dt=0.2,
    N=20,
    max_iter=2,
    conv_tol=0.08,
    solver=cp.CLARABEL,
):
    """Solves the Deterministic Nominal MPC problem against the mean predicted pedestrian trajectory.

    Parameters
    ----------
    x0 : float
        Current vehicle longitudinal position in meters.
    v0 : float
        Current vehicle longitudinal speed in m/s.
    scenarios : ndarray
        Predicted pedestrian trajectories of shape (D, S, N, 2) or (D, S, N).
    v_des : float, optional
        Target cruising speed under unconstrained conditions (default: 8.0 m/s).
    v_max, v_min : float, optional
        Vehicle speed bounds (default: 15.0, 0.0 m/s).
    c : float, optional
        Distance scaling parameter for target velocity (default: 20.0 m).
    w : float, optional
        Weight penalizing control acceleration effort (default: 0.2).
    a_min, a_max : float, optional
        Actuator acceleration bounds (default: -8.0, +3.0 m/s^2).
    road_y : float, optional
        Road centerline lateral coordinate (default: 20.0 m).
    dt : float, optional
        Sampling period in seconds (default: 0.2 s).
    N : int, optional
        Prediction lookahead horizon steps (default: 20 steps).
    max_iter : int, optional
        Maximum SCP iterations for self-consistency (default: 2).
    conv_tol : float, optional
        Position convergence threshold in meters (default: 0.08 m).
    solver : cp.Solver, optional
        Convex quadratic program solver (default: cp.CLARABEL).

    Returns
    -------
    a_opt : float or None
        Optimal initial acceleration command a_0*.
    info : dict
        Telemetry containing solver status, solve time, planned profiles, and nominal predictions.
    """
    t_start = time.perf_counter()

    # 1. Normalize scenario input shape to (D, S, N, features)
    scenarios_arr = np.asarray(scenarios, dtype=float)
    if scenarios_arr.ndim == 2:
        scenarios_arr = scenarios_arr[None, None, :, :]
    elif scenarios_arr.ndim == 3:
        if scenarios_arr.shape[-1] == 2:
            scenarios_arr = scenarios_arr[None, :, :, :]
        else:
            scenarios_arr = scenarios_arr[:, :, :, None]

    D = scenarios_arr.shape[0] if scenarios_arr.size > 0 else 0

    # 2. Collapse scenarios into a single nominal mean trajectory: shape (D, N, 2)
    if D > 0:
        mean_scenarios = np.mean(scenarios_arr, axis=1)  # shape: (D, N, features)
    else:
        mean_scenarios = None

    # 3. Warm-start planned vehicle trajectory guess
    t_steps = np.arange(1, N + 1) * dt
    planned_x = x0 + v0 * t_steps  # shape (N,)

    final_info = None
    final_a_opt = None
    iter_count = 0

    # 4. Iterative Sequential Convex Programming (SCP) Loop
    for iteration in range(max_iter):
        iter_count += 1

        # Evaluate distance against nominal mean trajectory
        if D > 0 and mean_scenarios is not None:
            if mean_scenarios.shape[-1] >= 2:
                ped_x = mean_scenarios[:, :, 0]          # (D, N)
                ped_y = mean_scenarios[:, :, 1]          # (D, N)
                dx = ped_x - planned_x[None, :]          # broadcasting: (D, N)
                dy = ped_y - road_y                      # (D, N)
                dist_all = np.sqrt(dx**2 + dy**2)        # (D, N)
                d_k = np.min(dist_all, axis=0)           # closest obstacle per step: (N,)
            else:
                ped_x = mean_scenarios[:, :, 0]
                dx = ped_x - planned_x[None, :]
                d_k = np.min(np.abs(dx), axis=0)         # (N,)

            # Desired velocity regulation along the single nominal trajectory
            v_desired = v_des * (d_k / (d_k + c))        # (N,)
        else:
            d_k = np.full(N, 100.0, dtype=float)
            v_desired = np.full(N, v_des, dtype=float)

        # 5. Decision Variables
        a = cp.Variable(N, name="a")
        v = cp.Variable(N + 1, name="v")
        x = cp.Variable(N + 1, name="x")

        # Double-integrator kinematic state transitions
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

        # 6. Objective: Single-Track Nominal Tracking Loss + Acceleration Regularization
        tracking_error = v[:N] - v_desired  # (N,)
        cost = cp.sum_squares(tracking_error) + w * cp.sum_squares(a)

        objective = cp.Minimize(cost)
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
            return None, {"status": problem.status, "iterations": iter_count}

        # Update planned trajectory and evaluate convergence
        new_planned_x = np.array(x.value[1:])
        pos_shift = float(np.max(np.abs(new_planned_x - planned_x)))
        planned_x = new_planned_x

        final_a_opt = float(a.value[0])
        final_info = {
            "status": problem.status,
            "objective": float(problem.value),
            "a_full": np.array(a.value),
            "v_full": np.array(v.value),
            "x_full": np.array(x.value),
            "v_desired": v_desired,
            "nominal_ped_dists": d_k,
            "iterations": iter_count,
            "pos_shift": pos_shift,
        }

        if pos_shift < conv_tol:
            break

    solve_time_ms = (time.perf_counter() - t_start) * 1000.0
    final_info["solve_time_ms"] = solve_time_ms

    return final_a_opt, final_info
