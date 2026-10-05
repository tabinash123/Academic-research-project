"""Real-Time SAA-MPC (Sample Average Approximation) Closed-Loop Simulation with Interactive Pygame Display.

Baseline non-robust optimization model for benchmarking against WDRO-MPC.
- Minimizes empirical expected loss across all S=100 historical scenario trajectories.
- Drops the Wasserstein ambiguity ball (no epsilon, no lambda, no ground transport matrix M).
- Retains identical live LiDAR sensing, pedestrian anchoring, and vehicle kinematics.
- Renders the forward planned acceleration profile in real time.
"""

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
from threat_assessment import predict_detected_pedestrians
from saa_mpc import solve_saa_mpc
from pygame_prediction_renderer import PygamePredictionRenderer

# ======================================================
# Setup Offline Data Artifacts (Full S=100 Scenarios)
# ======================================================

artifacts_dir = parent_dir / "past_artifacts"
res_file = artifacts_dir / "residuals.npy"

if not res_file.exists():
    raise FileNotFoundError(f"Missing required offline artifacts in {artifacts_dir}")

# Load full historical position residuals: (S=100, N=20, 2D)
residuals = np.load(res_file)[:, 1:21, :2]
S, N, _ = residuals.shape

# ======================================================
# Simulation Setup (Full Multi-Pedestrian Environment)
# ======================================================

env = Environment(num_pedestrians=5)
renderer = PygamePredictionRenderer(
    env,
    title=f"Real-Time SAA-MPC (Sample Average Approximation Baseline) | S={S} scenarios"
)

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
current_planned_accels = None

frame_idx = 0
paused = False
running = True

print("=" * 75)
print("  Real-Time Closed-Loop SAA-MPC Simulation (Normal Baseline Model)")
print("  - Formulation: Sample Average Approximation (Risk-Neutral Expected Loss)")
print("  - Target Cruising Speed: v_des = 8.0 m/s (~28.8 km/h)")
print("  - Scenario Pool: Full S = 100 historical residuals")
print("  - Dropped WDRO: No ambiguity ball, no lambda, no transport matrix M")
print("  - Controls: [SPACE] Pause/Resume | [R] Reset | [ESC] Exit")
print("=" * 75)

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
                solve_time_ms = 0.0
                current_scenarios = None
                current_detected = None
                current_ped_pos = None
                current_planned_accels = None

    # 2. Closed-Loop Perception, SAA-MPC Solve, & Physics Advance
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

            # Solve SAA-MPC with all 100 historical scenarios
            a_opt, info = solve_saa_mpc(
                x0=v.center[0],
                v0=v.speed_real,
                scenarios=current_scenarios,
                v_des=8.0,
                v_max=15.0,
                c=20.0,
                w=0.2,
                dt=DT,
                N=N,
                max_iter=2,
                conv_tol=0.08
            )
            solve_time_ms = info["solve_time_ms"]

            if a_opt is not None:
                current_accel = float(a_opt)
                current_planned_accels = info.get("a_full")
            else:
                current_accel = -8.0  # Failsafe emergency braking
                current_planned_accels = np.full(N, -8.0)
        else:
            # Clear roadway ahead -> smooth cruising toward v_des = 8.0 m/s
            speed_error = 8.0 - v.speed_real
            current_accel = float(np.clip(1.5 * speed_error, -3.0, 2.0))
            solve_time_ms = 0.0
            current_scenarios = None
            current_detected = None
            current_ped_pos = None
            current_planned_accels = None

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
        planned_accels=current_planned_accels,
        paused=paused
    )

    # 5. Maintain 5 Hz Physical Rate
    clock.tick(SIM_FPS)

renderer.close()
print("SAA simulation finished cleanly.")
