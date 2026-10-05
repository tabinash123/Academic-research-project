import numpy as np
from units import (
    real_speed_to_sim,
    sim_speed_to_real,
    real_accel_to_sim,
)


class Vehicle:
    """Ego vehicle with 2D position (x, y) and longitudinal + lateral kinematics.

    Physics is defined in real-world units (m/s, m/s^2) and clipped to
    a fixed physical operating envelope. Supports both 1D longitudinal
    acceleration (accel_real = float) and 2D steering/speed control
    (accel_real = (accel_x, accel_y)).
    """

    # Longitudinal Operating Envelope
    MIN_SPEED_REAL_X = 0.0  # m/s
    MAX_SPEED_REAL_X = 15.0  # m/s (~54 km/h)
    MIN_ACCEL_REAL_X = -8.0  # m/s^2 (braking)
    MAX_ACCEL_REAL_X = 3.0  # m/s^2 (acceleration)

    # Lateral Operating Envelope (Steering / Nudge)
    MAX_SPEED_REAL_Y = 2.0  # m/s (max lateral velocity)
    MAX_ACCEL_REAL_Y = 2.0  # m/s^2 (max lateral steering acceleration)

    def __init__(self, x, y, speed_real, length, width, world_width, road_y=20.0, road_width=12.0):
        self.x = float(x)
        self.y = float(y)
        self.length = float(length)
        self.width = float(width)
        self.world_width = float(world_width)
        self.road_y = float(road_y)
        self.road_width = float(road_width)

        # Longitudinal and lateral speeds
        speed_real_x = np.clip(speed_real, self.MIN_SPEED_REAL_X, self.MAX_SPEED_REAL_X)
        self.speed_sim_x = real_speed_to_sim(speed_real_x)
        self.speed_sim_y = 0.0

        self.last_accel_real_x = 0.0
        self.last_accel_real_y = 0.0

    @property
    def speed_sim(self):
        """Backward compatibility for speed_sim (longitudinal)."""
        return self.speed_sim_x

    @property
    def speed_real(self):
        """Current longitudinal speed in real-world units (m/s)."""
        return sim_speed_to_real(self.speed_sim_x)

    @property
    def speed_real_y(self):
        """Current lateral speed in real-world units (m/s)."""
        return sim_speed_to_real(self.speed_sim_y)

    @property
    def last_accel_real(self):
        """Backward compatibility for last_accel_real."""
        return self.last_accel_real_x

    def move(self, accel_real=0.0):
        """Advance one step.

        accel_real: can be a float (longitudinal accel_x) or a tuple/list (accel_x, accel_y).
        """
        if isinstance(accel_real, (tuple, list, np.ndarray)):
            accel_x = float(accel_real[0])
            accel_y = float(accel_real[1]) if len(accel_real) > 1 else 0.0
        else:
            accel_x = float(accel_real)
            accel_y = 0.0

        # Clip accelerations to physical envelopes
        accel_x = np.clip(accel_x, self.MIN_ACCEL_REAL_X, self.MAX_ACCEL_REAL_X)
        accel_y = np.clip(accel_y, -self.MAX_ACCEL_REAL_Y, self.MAX_ACCEL_REAL_Y)

        self.last_accel_real_x = accel_x
        self.last_accel_real_y = accel_y

        # Convert to sim-units
        accel_sim_x = real_accel_to_sim(accel_x)
        accel_sim_y = real_accel_to_sim(accel_y)

        # Update longitudinal speed
        min_vx_sim = real_speed_to_sim(self.MIN_SPEED_REAL_X)
        max_vx_sim = real_speed_to_sim(self.MAX_SPEED_REAL_X)
        self.speed_sim_x = np.clip(self.speed_sim_x + accel_sim_x, min_vx_sim, max_vx_sim)

        # Update lateral speed
        max_vy_sim = real_speed_to_sim(self.MAX_SPEED_REAL_Y)
        self.speed_sim_y = np.clip(self.speed_sim_y + accel_sim_y, -max_vy_sim, max_vy_sim)

        # Update positions
        self.x += self.speed_sim_x
        self.y += self.speed_sim_y

        # Keep vehicle inside physical road boundaries
        half_car_w = self.width / 2.0
        min_y = self.road_y - self.road_width / 2.0 + half_car_w
        max_y = self.road_y + self.road_width / 2.0 - half_car_w
        self.y = np.clip(self.y, min_y, max_y)

        # Wrap around world horizontally
        if self.x > self.world_width:
            self.x = -self.length
            self.y = self.road_y  # reset to center line on lap wrap
            self.speed_sim_y = 0.0

    @property
    def center(self):
        """(x, y) of vehicle center where LiDAR is mounted."""
        return (self.x + self.length / 2.0, self.y)
