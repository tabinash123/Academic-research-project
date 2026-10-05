import sys
import time
from pathlib import Path
import numpy as np

# Ensure UTF-8 output encoding for terminal
sys.stdout.reconfigure(encoding="utf-8")

# Ensure local directory is importable
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))

from scenario_generation import generate_scenarios
from wdro_mpc import solve_wdro_mpc

# Load precomputed offline data from past_artifacts
artifacts_dir = parent_dir / "past_artifacts"
res_file = artifacts_dir / "residuals.npy"
dist_file = artifacts_dir / "distance_matrix.npy"

if not res_file.exists() or not dist_file.exists():
    print("Error: Required offline artifacts not found in 'past_artifacts/'.")
    sys.exit(1)

# Slice to future position residuals (t=1..20, features x,y) -> (100, 20, 2)
residuals = np.load(res_file)[:, 1:21, :2]
dist_matrix = np.load(dist_file)

S, N, _ = residuals.shape

# ==============================================================
# Test Scenario: Pedestrian standing at x=35, vehicle at x=10
# Vehicle cruising at 8.0 m/s toward the pedestrian
# ==============================================================
ped_p0 = np.array([35.0, 20.0])  # Pedestrian position (m)
ped_v0 = np.array([0.0, 0.0])  # Standing still

veh_x0 = 10.0  # Vehicle x-position (m)
veh_v0 = 8.0  # Vehicle speed (m/s)

# Generate S=100 scenarios for this pedestrian
scenarios = generate_scenarios(ped_p0, ped_v0, residuals, horizon=N)  # (100, 20, 2)
scenarios_x_all = np.expand_dims(scenarios[:, :, 0], axis=0)  # (1, 100, 20)

print("=" * 60)
print("WDRO-MPC Controller Sanity & Performance Test")
print("=" * 60)
print(f"Vehicle State:       x0 = {veh_x0:.1f} m,  v0 = {veh_v0:.1f} m/s (v_des = 8.0 m/s)")
print(f"Pedestrian State:    x  = {ped_p0[0]:.1f} m,  v  = {ped_v0[0]:.1f} m/s")
print(f"Initial Gap:         {ped_p0[0] - veh_x0:.1f} m (g_min = 5.0 m)")
print(f"Prediction Horizon:  N = {N} steps (4.0 s lookahead at DT=0.2s)")
print(f"Scenarios Shape:     {scenarios_x_all.shape}")
print(f"Distance Matrix:     {dist_matrix.shape}")
print("-" * 60)

# Solve with timing
start_time = time.perf_counter()
a_opt, info = solve_wdro_mpc(
    x0=veh_x0,
    v0=veh_v0,
    scenarios_x_all=scenarios_x_all,
    dist_matrix=dist_matrix,
    v_des=8.0,
    g_min=5.0,
    epsilon=0.1,
    dt=0.2,
    N=N,
)
solve_time_ms = (time.perf_counter() - start_time) * 1000.0

print(f"Solver Status:       {info['status']}")
print(f"Solve Time:          {solve_time_ms:.2f} ms")
print(f"Objective Value:     {info['objective']:.4f}" if info['objective'] is not None else "Objective: N/A")

if a_opt is not None:
    print(f"\n>>> Optimal a[0] = {a_opt:+.4f} m/s²")
    if a_opt < -0.5:
        print("    [PASS] Vehicle is BRAKING safely to respect g_min.")
    elif a_opt > 0.5:
        print("    [ALERT] Vehicle is accelerating toward pedestrian.")
    else:
        print("    [INFO] Cruising / small adjustment.")

    a_full = info["a_full"]
    if a_full is not None:
        print(f"\nAcceleration Profile across horizon (N={N}):")
        print(" ".join([f"{a_full[t]:+.2f}" for t in range(min(10, len(a_full)))]))
else:
    print(f"\n>>> Solver Failed! Status: {info['status']}")

print("=" * 60)
