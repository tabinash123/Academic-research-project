import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from environment import Environment
from renderer import Renderer

# ======================================================
# Build the simulation
# ======================================================

env = Environment()
renderer = Renderer(env)

# Car's path is a circular queue: once it exits at world_width it
# wraps back to -car_length (see Vehicle.move()). Size total_frames
# to show exactly NUM_CYCLES full laps.
# Note: env.vehicle.speed_sim is the internal sim-units/frame speed
# (derived from the real-world car_speed_real via units.py) -- use
# this, not speed_real, since it's what actually advances env.vehicle.x.

NUM_CYCLES = 5
cycle_distance = env.world_width + env.vehicle.length
frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
total_frames = frames_per_cycle * NUM_CYCLES

# ======================================================
# Animation loop
# ======================================================
# This mirrors the standard simulator skeleton:
#   environment.step() -> lidar detection (done inside step()) -> render
# A controller would slot in between step() and render() later.


def update(frame):
    env.step()  # advances vehicle, pedestrians, and runs lidar detection
    current_cycle = frame // frames_per_cycle + 1
    return renderer.draw(current_cycle, NUM_CYCLES)


ani = FuncAnimation(
    renderer.fig,
    update,
    frames=total_frames,
    interval=100,
    blit=False
)

plt.show()
