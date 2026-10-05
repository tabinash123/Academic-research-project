import numpy as np
import pygame

from environment import Environment
from pygame_renderer import PygameRenderer

# ======================================================
# Simulation Setup
# ======================================================

env = Environment()
renderer = PygameRenderer(env, title="Real-Time Pedestrian-Aware Autonomous Driving Simulation")

NUM_CYCLES = 5
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

# Physical clock settings:
# 5 Hz simulation rate corresponds to 5 FPS real-world speed (0.2s physical step)
SIM_FPS = 5

clock = pygame.time.Clock()
frame_idx = 0
paused = False
running = True

print("=" * 65)
print("  Real-Time Pygame Simulation Started")
print("  - Press [SPACE]  : Pause / Resume simulation")
print("  - Press [R]      : Reset simulation")
print("  - Press [ESC]    : Exit simulation")
print("=" * 65)

# ======================================================
# Real-Time Event & Simulation Loop
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
                env = Environment()
                renderer.env = env
                frame_idx = 0

    # 2. Advance Physics when not paused
    if not paused:
        env.step()  # advances vehicle, pedestrians, and LiDAR detection
        frame_idx += 1

    current_cycle = frame_idx // frames_per_cycle + 1

    # 3. Render State Live to Screen (< 0.5 ms CPU cost)
    renderer.draw(
        current_cycle=min(current_cycle, NUM_CYCLES),
        total_cycles=NUM_CYCLES,
        paused=paused
    )

    # 4. Pace clock to physical 5 Hz time step
    clock.tick(SIM_FPS)

renderer.close()
print("Simulation finished cleanly.")
