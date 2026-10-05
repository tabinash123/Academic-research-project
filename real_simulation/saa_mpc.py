"""Sample Average Approximation (SAA) Model Predictive Control Solver.

Baseline non-robust optimization model for benchmarking against WDRO-MPC.
- Minimizes empirical expected loss across all S scenarios with uniform probability (1/S).
- Drops the Wasserstein ambiguity ball (no epsilon, no lambda, no ground transport matrix M).
- Retains identical double-integrator vehicle kinematics and distance-governed speed regulation.
- Uses Sequential Convex Programming (SCP) for self-consistent trajectory-distance coupling.
"""

import time
import numpy as np
import cvxpy as cp


def solve_saa_mpc(
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
    """Solves the SAA (Sample Average Approximation) risk-neutral MPC problem.

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
        Telemetry containing solver status, solve time, planned profiles, and objective value.
    """
    t_start = time.perf_counter()

    # 1. Normalize scenario input shape to (D, S, N, features)
    scenarios_arr = np.asarray(scenarios, dtype=float)
    if scenarios_arr.ndim == 2:
        scenarios_arr = scenarios_arr[None, :, :]
    elif scenarios_arr.ndim == 3 and scenarios_arr.shape[-1] == 2:
        scenarios_arr = scenarios_arr[None, :, :, :]

    D = scenarios_arr.shape[0] if scenarios_arr.size > 0 else 0
    S = scenarios_arr.shape[1] if D > 0 else 1

    # 2. Warm-start planned vehicle trajectory guess
    t_steps = np.arange(1, N + 1) * dt
    planned_x = x0 + v0 * t_steps  # shape (N,)

    final_info = None
    final_a_opt = None
    iter_count = 0

    # 3. Iterative Sequential Convex Programming (SCP) Loop
    for iteration in range(max_iter):
        iter_count += 1

        # Evaluate distance and distance-governed desired velocity
        if D > 0:
            if scenarios_arr.ndim == 4 and scenarios_arr.shape[-1] == 2:
                ped_x = scenarios_arr[:, :, :, 0]          # (D, S, N)
                ped_y = scenarios_arr[:, :, :, 1]          # (D, S, N)
                dx = ped_x - planned_x[None, None, :]     # broadcasting
                dy = ped_y - road_y
                dist_all = np.sqrt(dx**2 + dy**2)         # (D, S, N)
                d_k = np.min(dist_all, axis=0)            # closest hazard per scenario: (S, N)
            else:
                ped_x = scenarios_arr if scenarios_arr.ndim == 3 else scenarios_arr[:, :, :, 0]
                dx = ped_x - planned_x[None, None, :]
                d_k = np.min(np.abs(dx), axis=0)          # (S, N)

            # Desired velocity regulation per scenario (S, N)
            v_desired = v_des * (d_k / (d_k + c))
        else:
            v_desired = np.full((S, N), v_des, dtype=float)

        # 4. Decision Variables
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

        # 5. Objective: Empirical Mean Loss (Sample Average Approximation)
        # Note: No lambda, no s, no M transport matrix
        if D > 0:
            tracking_error = v[:N][None, :] - v_desired  # shape: (S, N)
            scenario_losses = cp.sum_squares(tracking_error, axis=1) + w * cp.sum_squares(a)  # (S,)
            cost = (1.0 / S) * cp.sum(scenario_losses)
        else:
            cost = cp.sum_squares(v[:N] - v_des) + w * cp.sum_squares(a)

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
            "iterations": iter_count,
            "pos_shift": pos_shift,
        }

        if pos_shift < conv_tol:
            break

    solve_time_ms = (time.perf_counter() - t_start) * 1000.0
    final_info["solve_time_ms"] = solve_time_ms

    return final_a_opt, final_info
