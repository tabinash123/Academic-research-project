"""Scenario Snapshot Extractor Module.

Harvests realistic pedestrian threat encounters from the decoupled simulation
(vehicle cruising at constant speed), computes nominal constant-velocity predictions,
anchors historical residuals, and packages distinct snapshot benchmarks into a
reusable data file for standalone mathematical model testing.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Setup search paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir / "environment_setup"))
from environment import Environment
from units import DT
from scenario_generation import nominal_rollout, generate_scenarios


def extract_snapshots(
    num_snapshots=4,
    total_simulation_frames=200,
    min_frame_separation=20,
    seed=42,
    output_path=None
):
    """Simulates the environment, captures distinct encounter snapshots,

    computes nominal trajectories, anchors historical residuals, and saves the data.

    Parameters
    ----------
    num_snapshots : int
        Number of distinct encounter timeframes to extract (default: 4).
    total_simulation_frames : int
        Number of frames to advance the constant-speed vehicle simulation (default: 200).
    min_frame_separation : int
        Minimum timesteps between selected snapshots to ensure diverse scenarios (default: 20 = 4.0s).
    seed : int
        Random seed for reproducibility.
    output_path : Path or str, optional
        Where to save the .npz data. Defaults to 'real_simulation/scenario_snapshots.npz'.

    Returns
    -------
    snapshots : list of dict
        Extracted snapshot data structures.
    """
    np.random.seed(seed)

    if output_path is None:
        output_path = current_dir / "scenario_snapshots.npz"
    else:
        output_path = Path(output_path)

    # 1. Load offline historical behavioral residuals (S=100, N=20, 2D)
    res_file = parent_dir / "past_artifacts" / "residuals.npy"
    if not res_file.exists():
        raise FileNotFoundError(f"Missing residual artifact at: {res_file}")

    residuals = np.load(res_file)[:, 1:21, :2]  # shape (100, 20, 2)
    S, N, _ = residuals.shape

    print("=" * 75)
    print("  SCENARIO SNAPSHOT EXTRACTION MODULE")
    print(f"  - Target Snapshots: {num_snapshots} distinct encounters")
    print(f"  - Lookahead Horizon: N = {N} steps ({N * DT:.1f} s)")
    print(f"  - Residual Pool: S = {S} historical trajectories")
    print(f"  - Vehicle Mode: Decoupled constant cruising (8.0 m/s, accel = 0.0)")
    print("=" * 75)

    # 2. Run Decoupled Environment and Log All Active Encounter Frames
    env = Environment(num_pedestrians=5, car_speed_real=8.0)
    candidate_encounters = []

    for frame in range(total_simulation_frames):
        # Step simulation with constant speed (accel = 0.0)
        detections = env.step(accel_real=0.0)
        v = env.vehicle

        if detections is not None and len(detections["indices"]) > 0:
            det_indices = np.array(detections["indices"])
            speed_real = env.pedestrians.speed_real
            heading = env.pedestrians.heading
            velocities = np.column_stack([
                speed_real * np.cos(heading),
                speed_real * np.sin(heading)
            ])

            ped_positions = env.pedestrians.positions[det_indices]
            ped_velocities = velocities[det_indices]

            candidate_encounters.append({
                "frame": frame,
                "sim_time": frame * DT,
                "vehicle_x": float(v.center[0]),
                "vehicle_y": float(v.center[1]),
                "vehicle_v": float(v.speed_real),
                "det_indices": det_indices,
                "ped_positions": ped_positions,
                "ped_velocities": ped_velocities,
            })

    total_candidates = len(candidate_encounters)
    if total_candidates == 0:
        raise RuntimeError("No pedestrian encounters detected during simulation. Try increasing simulation frames.")

    print(f"\n[1/3] Simulation finished. Logged {total_candidates} active LiDAR encounter frames.")

    # 3. Select Diverse, Non-Overlapping Snapshots
    # Filter candidates by enforcing min_frame_separation
    separated_candidates = []
    last_frame = -min_frame_separation

    # Shuffle candidates to pick diverse random moments
    permuted_indices = np.random.permutation(total_candidates)
    for idx in permuted_indices:
        cand = candidate_encounters[idx]
        if all(abs(cand["frame"] - sc["frame"]) >= min_frame_separation for sc in separated_candidates):
            separated_candidates.append(cand)
            if len(separated_candidates) == num_snapshots:
                break

    # If strict separation didn't yield enough, relax separation
    if len(separated_candidates) < num_snapshots:
        for idx in permuted_indices:
            cand = candidate_encounters[idx]
            if cand not in separated_candidates:
                separated_candidates.append(cand)
                if len(separated_candidates) == num_snapshots:
                    break

    # Sort selected snapshots chronologically
    separated_candidates.sort(key=lambda item: item["frame"])
    print(f"[2/3] Selected {len(separated_candidates)} distinct encounter timeframes.")

    # 4. Compute Nominal Rollout & Anchor Residuals for Each Snapshot
    snapshots = []
    save_dict = {
        "num_snapshots": len(separated_candidates),
        "horizon": N,
        "num_scenarios": S,
        "dt": DT,
    }

    for snap_idx, item in enumerate(separated_candidates):
        frame = item["frame"]
        sim_time = item["sim_time"]
        v_x = item["vehicle_x"]
        v_y = item["vehicle_y"]
        v_speed = item["vehicle_v"]
        det_indices = item["det_indices"]
        ped_pos = item["ped_positions"]
        ped_vel = item["ped_velocities"]
        D = len(det_indices)

        # For each detected pedestrian:
        # Step A: Nominal constant-velocity prediction over horizon N
        # Step B: Overlay residuals to generate S candidate scenarios
        nominal_trajectories = np.zeros((D, N, 2), dtype=float)
        scenarios = np.zeros((D, S, N, 2), dtype=float)

        for d in range(D):
            p0 = ped_pos[d]
            v0 = ped_vel[d]

            # Nominal Constant-Velocity Rollout: p_nom(k) = p0 + v0 * (k * dt)
            nom = nominal_rollout(p0, v0, horizon=N, dt=DT)
            nominal_trajectories[d] = nom

            # Residual Anchoring: scenarios(s, k) = nom(k) + residual(s, k)
            scen = generate_scenarios(p0, v0, residuals, horizon=N, dt=DT)
            scenarios[d] = scen

        scenarios_x = scenarios[:, :, :, 0]  # Longitudinal X coordinates: (D, S, N)

        snapshot_data = {
            "snapshot_id": snap_idx + 1,
            "frame": frame,
            "sim_time": sim_time,
            "vehicle_x": v_x,
            "vehicle_y": v_y,
            "vehicle_v": v_speed,
            "detected_count": D,
            "detected_indices": det_indices,
            "ped_positions": ped_pos,
            "ped_velocities": ped_vel,
            "nominal_trajectories": nominal_trajectories,
            "scenarios": scenarios,
            "scenarios_x": scenarios_x,
        }
        snapshots.append(snapshot_data)

        # Store in flat numpy save dict
        prefix = f"snap_{snap_idx}_"
        save_dict[f"{prefix}id"] = snap_idx + 1
        save_dict[f"{prefix}frame"] = frame
        save_dict[f"{prefix}sim_time"] = sim_time
        save_dict[f"{prefix}vehicle_x"] = v_x
        save_dict[f"{prefix}vehicle_y"] = v_y
        save_dict[f"{prefix}vehicle_v"] = v_speed
        save_dict[f"{prefix}detected_count"] = D
        save_dict[f"{prefix}detected_indices"] = det_indices
        save_dict[f"{prefix}ped_positions"] = ped_pos
        save_dict[f"{prefix}ped_velocities"] = ped_vel
        save_dict[f"{prefix}nominal"] = nominal_trajectories
        save_dict[f"{prefix}scenarios"] = scenarios
        save_dict[f"{prefix}scenarios_x"] = scenarios_x

    # 5. Save to .npz file
    np.savez_compressed(output_path, **save_dict)
    print(f"[3/3] Successfully saved {len(snapshots)} snapshots to: {output_path.name}")

    # 6. Display Formatted Summary Table
    print("\n" + "=" * 75)
    print(f"{'Snap ID':<8}{'Frame':<8}{'Time (s)':<10}{'Veh X (m)':<12}{'Ped Count':<12}{'Rel Gaps (m)'}")
    print("-" * 75)
    for s in snapshots:
        gaps = [f"{pos[0] - s['vehicle_x']:+.1f}" for pos in s["ped_positions"]]
        gap_str = ", ".join(gaps)
        print(f"#{s['snapshot_id']:<7}{s['frame']:<8}{s['sim_time']:<10.1f}{s['vehicle_x']:<12.1f}{s['detected_count']:<12}{gap_str}")
    print("=" * 75)

    return snapshots


def load_scenario_snapshots(filepath=None):
    """Clean loader utility to retrieve the scenario snapshots for mathematical modeling.

    Returns
    -------
    snapshots : list of dict
        Each dict contains:
        - 'snapshot_id' : int
        - 'frame' : int
        - 'sim_time' : float
        - 'vehicle_x' : float (x0)
        - 'vehicle_v' : float (v0)
        - 'detected_count' : int (D)
        - 'detected_indices' : ndarray (D,)
        - 'ped_positions' : ndarray (D, 2)
        - 'ped_velocities' : ndarray (D, 2)
        - 'nominal_trajectories' : ndarray (D, N, 2)
        - 'scenarios' : ndarray (D, S, N, 2)
        - 'scenarios_x' : ndarray (D, S, N)
    """
    if filepath is None:
        filepath = current_dir / "scenario_snapshots.npz"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Snapshot file not found: {filepath}. Run extract_snapshots() first.")

    data = np.load(filepath)
    num_snaps = int(data["num_snapshots"])
    snapshots = []

    for i in range(num_snaps):
        prefix = f"snap_{i}_"
        snapshots.append({
            "snapshot_id": int(data[f"{prefix}id"]),
            "frame": int(data[f"{prefix}frame"]),
            "sim_time": float(data[f"{prefix}sim_time"]),
            "vehicle_x": float(data[f"{prefix}vehicle_x"]),
            "vehicle_y": float(data[f"{prefix}vehicle_y"]),
            "vehicle_v": float(data[f"{prefix}vehicle_v"]),
            "detected_count": int(data[f"{prefix}detected_count"]),
            "detected_indices": data[f"{prefix}detected_indices"],
            "ped_positions": data[f"{prefix}ped_positions"],
            "ped_velocities": data[f"{prefix}ped_velocities"],
            "nominal_trajectories": data[f"{prefix}nominal"],
            "scenarios": data[f"{prefix}scenarios"],
            "scenarios_x": data[f"{prefix}scenarios_x"],
        })

    return snapshots


if __name__ == "__main__":
    # Execute extraction when run directly
    extract_snapshots(num_snapshots=4, total_simulation_frames=250, seed=42)
