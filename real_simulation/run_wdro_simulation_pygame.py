import sys
import time
from pathlib import Path
# pyrefly: ignore [missing-import]
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

# Load future position residuals (S=100, N=20, 2D) and distance matrix (100, 100)
residuals = np.load(res_file)[:, 1:21, :2]
dist_matrix = np.load(dist_file)
S, N, _ = residuals.shape

# ======================================================
# Simulation Setup (Full Multi-Pedestrian Environment)
# ======================================================

env = Environment(num_pedestrians=5)
renderer = PygamePredictionRenderer(env, title="Real-Time WDRO-MPC Autonomous Speed Control Simulation")

NUM_CYCLES = 5
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

# Physical clock pacing: 5 Hz physical rate (0.2s per step)
SIM_FPS = 5
clock = pygame.time.Clock()

current_accel = 0.0
solve_time_ms = 0.0
current_scenarios = None
current_detected = None
current_ped_pos = None

frame_idx = 0
paused = False
running = True

print("=" * 70)
print("  Real-Time Closed-Loop WDRO-MPC Simulation")
print("  - Unified Distance-Governed Speed Regulation")
print("  - Target Cruising Speed: v_des = 8.0 m/s (~28.8 km/h)")
print("  - Scenario Pool: Full S = 100 historical residuals")
print("  - Controls: [SPACE] Pause/Resume | [R] Reset | [ESC] Exit")
print("=" * 70)

# ======================================================
# Real-Time Closed-Loop Simulation Loop
# ======================================================

while running and frame_idx < total_frames:
    # 1. Process Window / Keyboard Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                env = Environment(num_pedestrians=5)
                renderer.env = env
                frame_idx = 0
                current_accel = 0.0
                current_scenarios = None
                current_detected = None

    # 2. Closed-Loop Perception, WDRO-MPC Solve, & Physics Advance
    if not paused:
        v = env.vehicle
        detections = env.detections

        # Check for pedestrians inside 12m LiDAR perception radius
        if detections is not None and len(detections["indices"]) > 0:
            speed_real = env.pedestrians.speed_real
            heading = env.pedestrians.heading
            velocities = np.column_stack([
                speed_real * np.cos(heading),
                speed_real * np.sin(heading)
            ])

            # Anchor future trajectory candidate scenarios
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

            # Live WDRO-MPC Solve using all 100 historical scenarios
            a_opt, info = solve_wdro_mpc(
                x0=v.center[0],
                v0=v.speed_real,
                scenarios=current_scenarios,
                dist_matrix=dist_matrix,
                v_des=8.0,
                v_max=15.0,
                c=15.0,
                w=0,
                epsilon=0.9,
                dt=DT,
                N=N
            )
            solve_time_ms = info["solve_time_ms"]

            if a_opt is not None:
                current_accel = float(a_opt)
            else:
                current_accel = -5.0  # Failsafe braking
        else:
            # Clear roadway ahead -> smooth cruising toward v_des = 8.0 m/s
            speed_error = 8.0 - v.speed_real
            current_accel = float(np.clip(1.5 * speed_error, -3.0, 2.0))
            solve_time_ms = 0.0
            current_scenarios = None
            current_detected = None
            current_ped_pos = None

        # Step the environment with controller's optimal acceleration
        env.step(accel_real=current_accel)
        frame_idx += 1

    # 3. Dynamic Vehicle Color Coding
    if current_accel < -0.3:
        renderer.COLOR_CAR = (220, 38, 38)      # Red when braking
    elif current_accel > 0.3:
        renderer.COLOR_CAR = (22, 163, 74)      # Green when accelerating
    else:
        renderer.COLOR_CAR = (29, 78, 216)      # Blue when cruising

    # 4. Render State Live to Display
    current_cycle = frame_idx // frames_per_cycle + 1
    detected_count = len(env.detections["indices"]) if env.detections is not None else 0

    renderer.draw(
        scenarios=current_scenarios,
        detected_indices=current_detected,
        ped_positions=current_ped_pos,
        current_cycle=min(current_cycle, NUM_CYCLES),
        total_cycles=NUM_CYCLES,
        detected_count=detected_count,
        current_accel=current_accel,
        solve_time_ms=solve_time_ms,
        paused=paused
    )

    # 5. Maintain 5 Hz Physical Rate
    clock.tick(SIM_FPS)

renderer.close()
print("Simulation finished cleanly.")
