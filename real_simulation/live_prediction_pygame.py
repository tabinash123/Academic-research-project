import sys
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
from threat_assessment import predict_detected_pedestrians
from pygame_prediction_renderer import PygamePredictionRenderer

# ======================================================
# Setup Simulation and Residual Artifacts
# ======================================================

artifacts_dir = parent_dir / "past_artifacts"
residuals_path = artifacts_dir / "residuals.npy"

if not residuals_path.exists():
    raise FileNotFoundError(f"Could not find residual dataset at: {residuals_path}")

# Load 100 offline residual samples across N=20 future steps (2D position)
residuals = np.load(residuals_path)[:, 1:21, :2]  # shape: (100, 20, 2)
S, N, _ = residuals.shape

env = Environment(num_pedestrians=5)
renderer = PygamePredictionRenderer(env, title="Real-Time DR-MPC Pedestrian Future Scenario Prediction")

NUM_CYCLES = 5
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

# Pacing: 5 Hz physical time step = 5 FPS real-time clock
SIM_FPS = 5

clock = pygame.time.Clock()
frame_idx = 0
paused = False
running = True

# Cached prediction output
current_scenarios = None
current_detected_indices = None
current_ped_positions = None
detected_count = 0

print("=" * 70)
print("  Real-Time DR-MPC Pedestrian Scenario Prediction Started")
print(f"  - Historical Residual Pool: S = {S} trajectories, Horizon N = {N} steps (4.0s)")
print("  - Keyboard Controls:")
print("      [SPACE] : Pause / Resume simulation")
print("      [R]     : Reset simulation")
print("      [ESC]   : Exit simulation")
print("=" * 70)

# ======================================================
# Real-Time Event and Simulation Loop
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
                current_scenarios = None
                current_detected_indices = None

    # 2. Advance Physics and Compute Live Predictions when not paused
    if not paused:
        env.step()  # advances vehicle, pedestrians, and LiDAR detection
        frame_idx += 1

        detections = env.detections
        if detections is not None and len(detections["indices"]) > 0:
            speed_real = env.pedestrians.speed_real
            heading = env.pedestrians.heading
            velocities = np.column_stack([
                speed_real * np.cos(heading),
                speed_real * np.sin(heading)
            ])

            # Anchor historical residuals to live detected pedestrian states
            result = predict_detected_pedestrians(
                vehicle_center=env.vehicle.center,
                detected_indices=detections["indices"],
                pedestrian_positions=env.pedestrians.positions,
                pedestrian_velocities=velocities,
                residuals=residuals,
                horizon=N
            )

            current_scenarios = result["scenarios"]  # (D, 100, 20, 2)
            current_detected_indices = result["detected_indices"]
            current_ped_positions = env.pedestrians.positions.copy()
            detected_count = result["num_detected"]
        else:
            current_scenarios = None
            current_detected_indices = None
            current_ped_positions = None
            detected_count = 0

    current_cycle = frame_idx // frames_per_cycle + 1

    # 3. Render State Live to Screen (< 1.5 ms CPU cost)
    renderer.draw(
        scenarios=current_scenarios,
        detected_indices=current_detected_indices,
        ped_positions=current_ped_positions,
        current_cycle=min(current_cycle, NUM_CYCLES),
        total_cycles=NUM_CYCLES,
        detected_count=detected_count,
        paused=paused
    )

    # 4. Pace clock to physical 5 Hz time step
    clock.tick(SIM_FPS)

renderer.close()
print("Simulation finished cleanly.")
