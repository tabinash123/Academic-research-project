"""Real-Time Closed-Loop WDRO-MPC Simulation with Chaotic Pedestrian Behaviors.

Rigorously stress-tests Distributionally Robust Model Predictive Control (WDRO-MPC)
against realistic behavioral entropy, including sudden road darting, panicking/hesitation,
180-degree mid-street direction flips, and erratic swerving.

Keyboard Controls:
- [SPACE] : Pause / Resume
- [R]     : Reset simulation environment
- [1]     : Set Chaos Level -> MILD
- [2]     : Set Chaos Level -> URBAN (default)
- [3]     : Set Chaos Level -> ADVERSARIAL (extreme stress test)
- [ESC]   : Exit cleanly
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

from units import DT
from threat_assessment import predict_detected_pedestrians
from wdro_mpc import solve_wdro_mpc
from chaotic_environment import ChaoticEnvironment
from chaotic_pedestrian import EVENT_LABELS
from pygame_prediction_renderer import PygamePredictionRenderer

# ======================================================
# Setup Offline Artifacts
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
# Simulation Setup (Chaotic Multi-Pedestrian Environment)
# ======================================================
current_preset = "URBAN"
env = ChaoticEnvironment(num_pedestrians=5, chaos_preset=current_preset)
renderer = PygamePredictionRenderer(
    env,
    title=f"Chaotic WDRO-MPC Stress Test [{current_preset}]"
)

NUM_CYCLES = 5
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

# Physical pacing at 5 Hz (dt = 0.2s)
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

print("=" * 75)
print("  Real-Time WDRO-MPC Chaotic Stress Test Simulation")
print("  - Unified Distance-Governed Speed Regulation: v_des = 8.0 m/s")
print(f"  - Initial Chaos Mode: [{current_preset}]")
print("  - Hotkeys:")
print("      [1] MILD Chaos | [2] URBAN Chaos | [3] ADVERSARIAL Stress-Test")
print("      [SPACE] Pause/Resume | [R] Reset | [ESC] Exit")
print("=" * 75)

# ======================================================
# Closed-Loop Simulation Loop
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
                env = ChaoticEnvironment(num_pedestrians=5, chaos_preset=current_preset)
                renderer.env = env
                frame_idx = 0
                current_accel = 0.0
                current_scenarios = None
                current_detected = None
                print(f"[Simulation Reset] Reset with [{current_preset}] chaos.")
            elif event.key == pygame.K_1:
                current_preset = "MILD"
                env.set_chaos_preset(current_preset)
                print(f"[Chaos Level] Switched to MILD.")
            elif event.key == pygame.K_2:
                current_preset = "URBAN"
                env.set_chaos_preset(current_preset)
                print(f"[Chaos Level] Switched to URBAN.")
            elif event.key == pygame.K_3:
                current_preset = "ADVERSARIAL"
                env.set_chaos_preset(current_preset)
                print(f"[Chaos Level] Switched to ADVERSARIAL (Extreme Stress Test)!")

    # 2. Closed-Loop Perception, Control, & Physics Advance
    if not paused:
        v = env.vehicle
        detections = env.detections

        # Check for pedestrians inside 12m LiDAR radius
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
                horizon=N,
            )

            current_scenarios = result["scenarios"]
            current_detected = result["detected_indices"]
            current_ped_pos = env.pedestrians.positions.copy()

            # Log any active chaotic states for detected pedestrians
            active_events = [
                f"Ped {idx+1}: {EVENT_LABELS.get(env.pedestrians.active_event[idx], 'UNKNOWN')} ({speed_real[idx]:.1f}m/s)"
                for idx in current_detected
                if env.pedestrians.active_event[idx] != 0
            ]
            if active_events:
                print(f"Frame {frame_idx:03d} | Chaotic Events: {', '.join(active_events)}")

            # Vectorized WDRO-MPC Solve
            a_opt, info = solve_wdro_mpc(
                x0=v.center[0],
                v0=v.speed_real,
                scenarios=current_scenarios,
                dist_matrix=dist_matrix,
                v_des=8.0,
                v_max=15.0,
                c=15.0,
                w=0.2,
                epsilon=0.5,
                dt=DT,
                N=N,
            )
            solve_time_ms = info["solve_time_ms"]

            if a_opt is not None:
                current_accel = float(a_opt)
            else:
                current_accel = -5.0  # Failsafe braking
        else:
            # Clear roadway ahead -> cruise toward v_des = 8.0 m/s
            speed_error = 8.0 - v.speed_real
            current_accel = float(np.clip(1.5 * speed_error, -3.0, 2.0))
            solve_time_ms = 0.0
            current_scenarios = None
            current_detected = None
            current_ped_pos = None

        # Step the environment with optimal acceleration
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
        paused=paused,
    )

    # 5. Maintain 5 Hz Physical Rate
    clock.tick(SIM_FPS)

pygame.quit()
print("Simulation finished cleanly.")
