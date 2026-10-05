import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class Renderer:
    """Pure visualization. Reads Environment state and draws it --
    contains no simulation physics of its own.

    Speed/acceleration labels are drawn using the agents' *_real
    properties, so what's shown on screen is always the true physical
    value (m/s, m/s^2) -- never the raw internal sim-unit number.
    """

    def __init__(self, env, figsize=(12, 4)):
        self.env = env

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, env.world_width)
        self.ax.set_ylim(0, env.world_height)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("#d8f0d0")

        # ---------------------------------
        # Static elements: drawn once, never touched again
        # ---------------------------------

        road = Rectangle(
            (0, env.road_y - env.road_width / 2),
            env.world_width, env.road_width, color="gray"
        )
        self.ax.add_patch(road)

        self.ax.plot(
            [0, env.world_width], [env.road_y, env.road_y],
            "--", color="white", linewidth=2
        )

        # ---------------------------------
        # Dynamic elements: created once, updated in-place each frame
        # ---------------------------------

        v = env.vehicle

        self.car_patch = Rectangle(
            (v.x, v.y - v.width / 2), v.length, v.width, color="blue"
        )
        self.ax.add_patch(self.car_patch)

        self.lidar_patch = Circle(
            v.center, env.lidar.radius,
            fill=False, linestyle="--", edgecolor="cyan", linewidth=2
        )
        self.ax.add_patch(self.lidar_patch)

        self.pedestrian_scatter = self.ax.scatter(
            env.pedestrians.x, env.pedestrians.y,
            s=200, c=["red"] * env.pedestrians.n,
            edgecolors="black", linewidths=0.5, zorder=3
        )

        # Speed/acceleration label above the car
        self.car_label = self.ax.text(
            v.x + v.length / 2, v.y + v.width / 2 + 1.5,
            "", ha="center", fontsize=8, color="darkblue", zorder=4
        )

        # Speed label above each pedestrian
        self.pedestrian_labels = [
            self.ax.text(
                env.pedestrians.x[i], env.pedestrians.y[i] + 1.2,
                "", ha="center", fontsize=6, color="black", zorder=4
            )
            for i in range(env.pedestrians.n)
        ]

        self.title = self.ax.set_title(
            "Simple Autonomous Driving Environment | Detected Pedestrians: 0"
        )

    def draw(self, current_cycle=1, total_cycles=1):
        """Update all artists from the current environment state.
        Call this AFTER env.step() has run.
        """

        env = self.env
        v = env.vehicle

        self.car_patch.set_xy((v.x, v.y - v.width / 2))
        self.lidar_patch.set_center(v.center)

        self.car_label.set_position((v.x + v.length / 2, v.y + v.width / 2 + 1.5))
        self.car_label.set_text(
            f"v={v.speed_real:.1f} m/s | a={v.last_accel_real:.1f} m/s\u00b2"
        )

        self.pedestrian_scatter.set_offsets(env.pedestrians.positions)

        ped_speed_real = env.pedestrians.speed_real
        for i, label in enumerate(self.pedestrian_labels):
            label.set_position((env.pedestrians.x[i], env.pedestrians.y[i] + 1.2))
            label.set_text(f"{ped_speed_real[i]:.1f} m/s")

        detections = env.detections
        if detections is not None:
            colors = np.where(detections["mask"], "lime", "red")
            detected_count = len(detections["indices"])
        else:
            colors = ["red"] * env.pedestrians.n
            detected_count = 0

        self.pedestrian_scatter.set_color(colors)

        self.title.set_text(
            f"Simple Autonomous Driving Environment | Cycle: {current_cycle}/{total_cycles} | "
            f"Detected Pedestrians: {detected_count}"
        )

        return (
            [self.car_patch, self.lidar_patch, self.title,
             self.pedestrian_scatter, self.car_label]
            + self.pedestrian_labels
        )
