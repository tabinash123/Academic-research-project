"""
Shared conversion between real-world physical units and the
simulation's internal units.

SCALE:  sim-units per meter (spatial). 1.0 means 1 sim-unit == 1 meter.
DT:     real-world seconds represented by one simulation step/frame.

Every agent (Vehicle, PedestrianGroup) computes its physics in real
units (m/s, m/s^2) and then converts to sim-units through these
functions -- this is the one place that scale/timestep live, so
changing the "zoom level" or step duration of the whole simulation
only ever requires editing this file.
"""

SCALE = 1.0   # sim-units per meter
DT = 0.2      # seconds per simulation step (5 Hz)


def real_speed_to_sim(v_real):
    """Real-world speed (m/s) -> sim-units moved per frame."""
    return v_real * SCALE * DT


def sim_speed_to_real(v_sim):
    """Sim-units-per-frame speed -> real-world speed (m/s). Used for display."""
    return v_sim / (SCALE * DT)


def real_accel_to_sim(a_real):
    """Real-world acceleration (m/s^2) -> change in sim-speed per frame."""
    return a_real * SCALE * (DT ** 2)


def sim_accel_to_real(a_sim):
    """Sim per-frame speed-change -> real-world acceleration (m/s^2). For display."""
    return a_sim / (SCALE * (DT ** 2))
