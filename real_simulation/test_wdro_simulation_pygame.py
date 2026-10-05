import sys
import time
from pathlib import Path
import numpy as np
import pygame

# Set up module search paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir / "environment_setup"))
sys.path.insert(0, str(parent_dir / "past_artifacts"))

from environment import Environment
from units import DT
from scenario_generation import generate_scenarios
from threat_assessment import predict_detected_pedestrians
from wdro_mpc import solve_wdro_mpc
from pygame_prediction_renderer import PygamePredictionRenderer

# ======================================================
# Setup Offline Data Artifacts
# ======================================================

artifacts_dir = parent_dir / "past_artifacts"
res_file = artifacts_dir / "residuals.npy"
dist_file = artifacts_dir / "distance_matrix.npy"

if not res_file.exists() or not dist_file.exists():
    raise FileNotFoundError(f"Missing required offline artifacts in {artifacts_dir}")

residuals = np.load(res_file)[:, 1:21, :2]  # (100, 20, 2)
dist_matrix = np.load(dist_file)           # (100, 100)
S, N, _ = residuals.shape

# ======================================================
# Initialize Benchmark Test Scenario
# ======================================================

def create_test_environment():
    """Builds a realistic dynamic urban street-crossing scenario:

    - Vehicle starts at x=5m cruising at nominal v=8.0 m/s
    - Pedestrian 1: Street-crosser starting at lower sidewalk (x=38m, y=10m) crossing across the road
    - Pedestrian 2: Walker on upper sidewalk (x=65m, y=28m)
    - Pedestrian 3: Street-crosser starting at upper sidewalk (x=90m, y=30m) crossing downward
    """
    env = Environment(
        num_pedestrians=3,
        car_x=5.0,
        car_speed_real=8.0,
        lidar_radius=12.0
    )
    from units import real_speed_to_sim

    # Pedestrian 1: Natural street crosser (crossing upward: South -> North)
    env.pedestrians.x[0] = 38.0
    env.pedestrians.y[0] = 11.0
    env.pedestrians.heading[0] = np.pi / 2.0  # 90 degrees (crossing road upward)
    env.pedestrians.speed_sim[0] = real_speed_to_sim(1.3)  # 1.3 m/s walking speed
    env.pedestrians.base_speed_sim[0] = real_speed_to_sim(1.3)

    # Pedestrian 2: Sidewalk walker along top
    env.pedestrians.x[1] = 65.0
    env.pedestrians.y[1] = 28.0
    env.pedestrians.heading[1] = 0.0          # 0 degrees (walking along sidewalk)
    env.pedestrians.speed_sim[1] = real_speed_to_sim(1.1)
    env.pedestrians.base_speed_sim[1] = real_speed_to_sim(1.1)

    # Pedestrian 3: Street crosser (crossing downward: North -> South)
    env.pedestrians.x[2] = 90.0
    env.pedestrians.y[2] = 29.0
    env.pedestrians.heading[2] = -np.pi / 2.0 # -90 degrees (crossing road downward)
    env.pedestrians.speed_sim[2] = real_speed_to_sim(1.4)
    env.pedestrians.base_speed_sim[2] = real_speed_to_sim(1.4)

    return env


env = create_test_environment()
renderer = PygamePredictionRenderer(env, title="WDRO-MPC Controller Live Test Simulation")

clock = pygame.time.Clock()
SIM_FPS = 5  # 5 Hz physical rate (0.2s per step)

current_accel = 0.0
solve_time_ms = 0.0
solver_status = "IDLE"
current_scenarios = None
current_detected = None
current_ped_pos = None

paused = False
running = True
step_count = 0
max_steps = 100

print("=" * 70)
print("  WDRO-MPC Closed-Loop Benchmark Test Simulation")
print(f"  - Initial Setup: Vehicle at x=10m (v=8.0 m/s) -> Pedestrian at x=35m")
print(f"  - Safety Distance Floor: g_min = 5.0 m")
print(f"  - Horizon: N={N} steps (4.0s), Scenarios S={S}")
print("  - Controls: [SPACE] Pause/Resume | [R] Restart Test | [ESC] Exit")
print("=" * 70)

# ======================================================
# Real-Time Closed-Loop Test Simulation Loop
# ======================================================

while running and step_count < max_steps:
    # 1. Handle Window / Keyboard Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                env = create_test_environment()
                renderer.env = env
                step_count = 0
                current_accel = 0.0
                current_scenarios = None
                current_detected = None

    # 2. Closed-Loop Perception & WDRO-MPC Optimization
    if not paused:
        v = env.vehicle
        ped_x = env.pedestrians.x[0]
        gap = ped_x - (v.x + v.length)

        # Perception: Check LiDAR detection
        detections = env.detections
        if detections is not None and len(detections["indices"]) > 0:
            speed_real = env.pedestrians.speed_real
            heading = env.pedestrians.heading
            velocities = np.column_stack([
                speed_real * np.cos(heading),
                speed_real * np.sin(heading)
            ])

            # Anchor future trajectory scenarios
            result = predict_detected_pedestrians(
                vehicle_center=v.center,
                detected_indices=detections["indices"],
                pedestrian_positions=env.pedestrians.positions,
                pedestrian_velocities=velocities,
                residuals=residuals,
                horizon=N
            )

            current_scenarios = result["scenarios"]
            current_detected = result["detected_indices"]
            current_ped_pos = env.pedestrians.positions.copy()
            scenarios_x_all = current_scenarios[:, :, :, 0]  # (D, 100, 20)

            # Solve Wasserstein DR-MPC Optimization
            t_solve_start = time.perf_counter()
            a_opt, info = solve_wdro_mpc(
                x0=v.center[0],
                v0=v.speed_real,
                scenarios_x_all=scenarios_x_all,
                dist_matrix=dist_matrix,
                v_des=8.0,
                g_min=5.0,
                epsilon=0.1,
                dt=DT,
                N=N
            )
            solve_time_ms = (time.perf_counter() - t_solve_start) * 1000.0

            if a_opt is not None:
                current_accel = float(a_opt)
                solver_status = info.get("status", "optimal")
            else:
                current_accel = -5.0  # Failsafe maximum braking if solver fails
                solver_status = "FAILED"
        else:
            # Clear road ahead -> Cruise at nominal speed
            current_accel = 0.0
            solver_status = "CRUISING (CLEAR)"
            current_scenarios = None
            current_detected = None
            current_ped_pos = None

        # Apply optimal acceleration to the vehicle
        env.step(accel_real=current_accel)
        step_count += 1

    # 3. Dynamic Vehicle Color Coding
    if current_accel < -0.5:
        renderer.COLOR_CAR = (220, 38, 38)      # Red when braking
    elif current_accel > 0.5:
        renderer.COLOR_CAR = (22, 163, 74)      # Green when accelerating
    else:
        renderer.COLOR_CAR = (29, 78, 216)      # Blue when cruising

    # 4. Render Frame Live to Screen
    v = env.vehicle
    detected_count = len(env.detections["indices"]) if env.detections is not None else 0

    renderer.draw(
        scenarios=current_scenarios,
        detected_indices=current_detected,
        ped_positions=current_ped_pos,
        current_cycle=step_count,
        total_cycles=max_steps,
        detected_count=detected_count,
        paused=paused
    )

    # 5. Maintain 5 Hz Physical Time Step
    clock.tick(SIM_FPS)

renderer.close()
print("\nTest simulation finished cleanly.")
