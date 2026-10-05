"""Head-to-Head Benchmark: WDRO-MPC vs. SAA-MPC (Sample Average Approximation).

Scientific verification of the value of Distributionally Robust Optimization:
- Evaluates both controllers under identical random seeds and pedestrian behaviors.
- WDRO-MPC: Optimizes against the worst-case distribution within a Wasserstein ball (epsilon=0.7).
- SAA-MPC: Optimizes against the empirical average distribution (uniform probability 1/S).
- Compares: Minimum Clearance Distance, Braking Deceleration, Jerk / Ride Comfort, and Solve Latency.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding for terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir / "environment_setup"))
sys.path.insert(0, str(parent_dir / "past_artifacts"))

from environment import Environment
from units import DT
from threat_assessment import predict_detected_pedestrians
from wdro_mpc_circular import solve_wdro_mpc_circular
from saa_mpc import solve_saa_mpc
from extract_scenario_snapshots import load_scenario_snapshots


def run_benchmark_sim(controller_type="wdro", epsilon=0.7, seed=42, total_frames=120, c=20.0, w=0.2, N=20):
    """Runs a standardized closed-loop simulation under identical conditions."""
    np.random.seed(seed)

    artifacts_dir = parent_dir / "past_artifacts"
    residuals = np.load(artifacts_dir / "residuals.npy")[:, 1:21, :2]
    dist_matrix = np.load(artifacts_dir / "distance_matrix.npy")
    S = residuals.shape[0]

    # For fast real-time solving in benchmark: S=25 subsampled or full 100
    S_BENCH = 25
    sub_idx = np.linspace(0, S - 1, S_BENCH, dtype=int)
    dist_matrix_bench = dist_matrix[np.ix_(sub_idx, sub_idx)]

    env = Environment(num_pedestrians=5)

    speeds = []
    accels = []
    min_ped_distances = []
    solve_times = []
    jerks = []
    encounter_accels = []

    last_accel = 0.0
    current_accel = 0.0

    for frame in range(total_frames):
        v = env.vehicle
        p_pos = env.pedestrians.positions

        # Calculate true physical distance to all pedestrians
        dx = p_pos[:, 0] - v.center[0]
        dy = p_pos[:, 1] - v.center[1]
        all_dists = np.sqrt(dx**2 + dy**2)
        min_dist = float(np.min(all_dists))
        min_ped_distances.append(min_dist)

        detections = env.detections

        if detections is not None and len(detections["indices"]) > 0:
            speed_real = env.pedestrians.speed_real
            heading = env.pedestrians.heading
            velocities = np.column_stack([
                speed_real * np.cos(heading),
                speed_real * np.sin(heading)
            ])

            result = predict_detected_pedestrians(
                vehicle_center=v.center,
                detected_indices=detections["indices"],
                pedestrian_positions=p_pos,
                pedestrian_velocities=velocities,
                residuals=residuals,
                horizon=N
            )

            scenarios_bench = result["scenarios"][:, sub_idx, :, :]

            if controller_type == "wdro":
                a_opt, info = solve_wdro_mpc_circular(
                    x0=v.center[0],
                    v0=v.speed_real,
                    scenarios=scenarios_bench,
                    dist_matrix=dist_matrix_bench,
                    v_des=8.0,
                    v_max=15.0,
                    c=c,
                    w=w,
                    epsilon=epsilon,
                    dt=DT,
                    N=N,
                    max_iter=2,
                    conv_tol=0.08
                )
            elif controller_type == "saa":
                a_opt, info = solve_saa_mpc(
                    x0=v.center[0],
                    v0=v.speed_real,
                    scenarios=scenarios_bench,
                    v_des=8.0,
                    v_max=15.0,
                    c=c,
                    w=w,
                    dt=DT,
                    N=N,
                    max_iter=2,
                    conv_tol=0.08
                )
            else:
                raise ValueError(f"Unknown controller: {controller_type}")

            solve_times.append(info.get("solve_time_ms", 0.0))

            if a_opt is not None:
                current_accel = float(a_opt)
            else:
                current_accel = -8.0

            encounter_accels.append(current_accel)
        else:
            speed_error = 8.0 - v.speed_real
            current_accel = float(np.clip(1.5 * speed_error, -3.0, 2.0))

        if frame > 0:
            jerk = (current_accel - last_accel) / DT
            jerks.append(jerk)
        last_accel = current_accel

        speeds.append(float(v.speed_real))
        accels.append(current_accel)

        env.step(accel_real=current_accel)

    accels_arr = np.array(accels)
    jerks_arr = np.array(jerks) if len(jerks) > 0 else np.array([0.0])
    encounter_accels_arr = np.array(encounter_accels) if len(encounter_accels) > 0 else np.array([0.0])

    return {
        "controller": controller_type.upper(),
        "min_clearance": float(np.min(min_ped_distances)),
        "peak_braking": float(np.min(accels_arr)),
        "mean_encounter_accel": float(np.mean(encounter_accels_arr)),
        "jerk_rms": float(np.sqrt(np.mean(jerks_arr**2))),
        "hard_braking_frames": int(np.sum(accels_arr < -3.0)),
        "min_speed": float(np.min(speeds)),
        "mean_speed": float(np.mean(speeds)),
        "avg_solve_ms": float(np.mean(solve_times)) if len(solve_times) > 0 else 0.0,
        "time_series": {
            "accels": accels,
            "speeds": speeds,
            "distances": min_ped_distances,
        }
    }


def compare_snapshots():
    """Compares WDRO vs SAA planned acceleration profiles on encounter snapshots."""
    snapshots = load_scenario_snapshots()
    artifacts_dir = parent_dir / "past_artifacts"
    dist_matrix = np.load(artifacts_dir / "distance_matrix.npy")
    sub_idx = np.linspace(0, dist_matrix.shape[0] - 1, 25, dtype=int)
    dist_matrix_bench = dist_matrix[np.ix_(sub_idx, sub_idx)]

    comparison = []
    for snap in snapshots:
        snap_id = snap["snapshot_id"]
        v_x = snap["vehicle_x"]
        v_speed = snap["vehicle_v"]
        scenarios_bench = snap["scenarios"][:, sub_idx, :, :]

        # WDRO (epsilon = 0.7)
        a_wdro, info_wdro = solve_wdro_mpc_circular(
            x0=v_x, v0=v_speed, scenarios=scenarios_bench, dist_matrix=dist_matrix_bench,
            v_des=8.0, c=20.0, w=0.2, epsilon=0.7, dt=DT, N=20
        )

        # SAA
        a_saa, info_saa = solve_saa_mpc(
            x0=v_x, v0=v_speed, scenarios=scenarios_bench,
            v_des=8.0, c=20.0, w=0.2, dt=DT, N=20
        )

        comparison.append({
            "snapshot_id": snap_id,
            "wdro_a0": float(a_wdro),
            "saa_a0": float(a_saa),
            "diff_a0": float(a_wdro - a_saa),
            "wdro_a_full": info_wdro.get("a_full", []).tolist() if info_wdro.get("a_full") is not None else [],
            "saa_a_full": info_saa.get("a_full", []).tolist() if info_saa.get("a_full") is not None else [],
            "wdro_latency_ms": info_wdro.get("solve_time_ms", 0.0),
            "saa_latency_ms": info_saa.get("solve_time_ms", 0.0),
        })
    return comparison


def main():
    print("=" * 90)
    print("  HEAD-TO-HEAD BENCHMARK: WDRO-MPC vs. SAA-MPC (NORMAL BASELINE)")
    print("=" * 90)
    print("Running closed-loop simulation under identical pedestrian movements...\n")

    wdro_res = run_benchmark_sim(controller_type="wdro", epsilon=0.7, seed=42)
    saa_res = run_benchmark_sim(controller_type="saa", seed=42)

    print("=" * 90)
    print(f"{'Performance Metric':<32} {'WDRO-MPC (Robust)':<24} {'SAA-MPC (Normal)':<24} {'Advantage'}")
    print("-" * 90)
    print(f"{'Minimum Clearance Buffer':<32} {wdro_res['min_clearance']:>6.2f} m {'':<16} {saa_res['min_clearance']:>6.2f} m {'':<16} {'WDRO safer' if wdro_res['min_clearance'] > saa_res['min_clearance'] else 'SAA closer'}")
    print(f"{'Peak Braking Deceleration':<32} {wdro_res['peak_braking']:>+6.2f} m/s² {'':<15} {saa_res['peak_braking']:>+6.2f} m/s²")
    print(f"{'Mean Encounter Deceleration':<32} {wdro_res['mean_encounter_accel']:>+6.2f} m/s² {'':<15} {saa_res['mean_encounter_accel']:>+6.2f} m/s² {'':<15} {'WDRO hedges' if abs(wdro_res['mean_encounter_accel']) > abs(saa_res['mean_encounter_accel']) else 'SAA gentle'}")
    print(f"{'Hard Braking Steps (a < -3)':<32} {wdro_res['hard_braking_frames']:>6d} frames {'':<14} {saa_res['hard_braking_frames']:>6d} frames")
    print(f"{'Minimum Speed in Encounters':<32} {wdro_res['min_speed']:>6.2f} m/s {'':<16} {saa_res['min_speed']:>6.2f} m/s")
    print(f"{'Average Cruising Speed':<32} {wdro_res['mean_speed']:>6.2f} m/s {'':<16} {saa_res['mean_speed']:>6.2f} m/s")
    print(f"{'Ride Comfort (Jerk RMS)':<32} {wdro_res['jerk_rms']:>6.2f} m/s³ {'':<15} {saa_res['jerk_rms']:>6.2f} m/s³")
    print(f"{'Average Solve Latency':<32} {wdro_res['avg_solve_ms']:>6.2f} ms {'':<17} {saa_res['avg_solve_ms']:>6.2f} ms {'':<17} {'SAA faster'}")
    print("-" * 90)

    # Snapshot Detailed Comparison
    print("\n--> Snapshot Decision Comparison (Immediate a0* Planned):")
    snap_data = compare_snapshots()
    print(f"{'Snapshot':<10} {'WDRO a0*':<16} {'SAA a0*':<16} {'Braking Difference (WDRO - SAA)':<32}")
    print("-" * 75)
    for s in snap_data:
        diff_str = f"{s['diff_a0']:+.3f} m/s² ({'WDRO brakes harder' if s['diff_a0'] < -0.01 else ('SAA brakes harder' if s['diff_a0'] > 0.01 else 'Identical')})"
        print(f"Snap #{s['snapshot_id']:<6} {s['wdro_a0']:>+8.3f} m/s²     {s['saa_a0']:>+8.3f} m/s²     {diff_str}")

    # Plot Comparison Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    time_axis = np.arange(120) * DT

    # 1. Acceleration Profile
    ax = axes[0, 0]
    ax.plot(time_axis, wdro_res["time_series"]["accels"], label="WDRO-MPC (Robust)", color="navy", linewidth=2.0)
    ax.plot(time_axis, saa_res["time_series"]["accels"], label="SAA-MPC (Normal)", color="crimson", linestyle="--", linewidth=1.8)
    ax.set_title("Applied Acceleration $a(t)$ over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Acceleration ($m/s^2$)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # 2. Vehicle Speed
    ax = axes[0, 1]
    ax.plot(time_axis, wdro_res["time_series"]["speeds"], label="WDRO-MPC (Robust)", color="navy", linewidth=2.0)
    ax.plot(time_axis, saa_res["time_series"]["speeds"], label="SAA-MPC (Normal)", color="crimson", linestyle="--", linewidth=1.8)
    ax.set_title("Vehicle Velocity $v(t)$ over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Speed ($m/s$)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # 3. Minimum Pedestrian Distance
    ax = axes[1, 0]
    ax.plot(time_axis, wdro_res["time_series"]["distances"], label="WDRO-MPC (Robust)", color="navy", linewidth=2.0)
    ax.plot(time_axis, saa_res["time_series"]["distances"], label="SAA-MPC (Normal)", color="crimson", linestyle="--", linewidth=1.8)
    ax.axhline(5.0, color="gray", linestyle=":", label="Safety Buffer (5m)")
    ax.set_title("Minimum Clearance to Pedestrians", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Distance ($m$)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # 4. Planned Horizon Acceleration in Snapshot #3
    ax = axes[1, 1]
    snap3 = snap_data[2]  # Snapshot #3
    horizon_steps = np.arange(len(snap3["wdro_a_full"]))
    ax.plot(horizon_steps, snap3["wdro_a_full"], "o-", label="WDRO-MPC Plan", color="navy", linewidth=2.0)
    ax.plot(horizon_steps, snap3["saa_a_full"], "s--", label="SAA-MPC Plan", color="crimson", linewidth=1.8)
    ax.set_title("Planned Horizon Accelerations $a_k$ (Snapshot #3)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lookahead Step $k$ ($0.2s$ per step)", fontsize=11)
    ax.set_ylabel("Planned Accel ($m/s^2$)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot_path = current_dir / "wdro_vs_saa_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[SAVED] Benchmark plot saved to: {plot_path}")

    # Save JSON summary
    out_json = current_dir / "wdro_vs_saa_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "wdro": {k: v for k, v in wdro_res.items() if k != "time_series"},
            "saa": {k: v for k, v in saa_res.items() if k != "time_series"},
            "snapshot_comparison": snap_data
        }, f, indent=2)
    print(f"[SAVED] Results saved to: {out_json}")


if __name__ == "__main__":
    main()
