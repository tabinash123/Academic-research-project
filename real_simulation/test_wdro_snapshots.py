"""Snapshot Benchmark Test for Official WDRO-MPC Solver.

Loads scenario_snapshots.npz, feeds each snapshot into the new
official WDRO-MPC formulation in wdro_mpc.py, and displays:
- Solver optimality status & latency (ms)
- Optimal initial acceleration a_0*
- Planned velocity profile v_0, ..., v_N
- Dual variables (lambda, mean epigraph s)
"""

import sys
from pathlib import Path
import numpy as np

# Setup search paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))

from extract_scenario_snapshots import load_scenario_snapshots
from wdro_mpc import solve_wdro_mpc


def run_snapshot_tests(snapshot_idx=None):
    # 1. Load distance matrix
    dist_file = parent_dir / "past_artifacts" / "distance_matrix.npy"
    if not dist_file.exists():
        raise FileNotFoundError(f"Distance matrix not found: {dist_file}")
    dist_matrix = np.load(dist_file)

    # 2. Load snapshots
    snapshots = load_scenario_snapshots()
    print("=" * 80)
    print("  OFFICIAL WDRO-MPC SOLVER SNAPSHOT BENCHMARK")
    print(f"  - Total Snapshots Loaded: {len(snapshots)}")
    print(f"  - Distance Matrix Shape: {dist_matrix.shape}")
    print("=" * 80)

    test_indices = [snapshot_idx - 1] if snapshot_idx is not None else range(len(snapshots))

    for idx in test_indices:
        snap = snapshots[idx]
        snap_id = snap["snapshot_id"]
        v_x = snap["vehicle_x"]
        v_speed = snap["vehicle_v"]
        D = snap["detected_count"]
        scenarios = snap["scenarios"]  # (D, S, N, 2)

        print(f"\n>>> Running Snapshot #{snap_id} (t = {snap['sim_time']:.1f}s | Frame {snap['frame']})")
        print(f"    Vehicle: x = {v_x:.1f} m, v = {v_speed:.1f} m/s | Detected Pedestrians: {D}")
        for d in range(D):
            p0 = snap["ped_positions"][d]
            print(f"    - Pedestrian {d+1}: pos = ({p0[0]:.1f}, {p0[1]:.1f}) m | rel_x = {p0[0] - v_x:+.1f} m")

        # Solve WDRO-MPC
        a_opt, info = solve_wdro_mpc(
            x0=v_x,
            v0=v_speed,
            scenarios=scenarios,
            dist_matrix=dist_matrix,
            v_des=8.0,
            v_max=15.0,
            c=5.0,
            w=0.5,
            epsilon=0.1,
            dt=0.2,
            N=20
        )

        print("-" * 50)
        print(f"    Solver Status    : {info['status']}")
        print(f"    Solve Latency    : {info['solve_time_ms']:.2f} ms")
        print(f"    Optimal Accel a0*: {a_opt:+.3f} m/s^2")
        print(f"    Dual Multiplier (lambda): {info['lam']:.4f}" if info['lam'] is not None else "    Dual Multiplier (lambda): N/A")
        print(f"    Mean Epigraph s  : {info['s_mean']:.4f}" if info['s_mean'] is not None else "    Mean Epigraph s  : N/A")
        print(f"    Objective Value  : {info['objective']:.4f}" if info['objective'] is not None else "    Objective Value  : N/A")

        if info["v_full"] is not None:
            v_traj = info["v_full"]
            print(f"    Planned Speeds   : start={v_traj[0]:.1f} m/s -> mid={v_traj[10]:.1f} m/s -> end={v_traj[-1]:.1f} m/s")

    print("\n" + "=" * 80)
    print("  ALL SNAPSHOT TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    # Test all 4 snapshots
    run_snapshot_tests()
