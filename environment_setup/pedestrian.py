import numpy as np
from dataclasses import dataclass, field
from units import (
    real_speed_to_sim,
    sim_speed_to_real,
    real_accel_to_sim,
)


# ======================================================
# Event type constants
# ======================================================

EVENT_NONE = 0
EVENT_SUDDEN_STOP = 1
EVENT_DIRECTION_FLIP = 2
EVENT_SPEED_BURST = 3
EVENT_LATERAL_JERK = 4
EVENT_HESITATION = 5


# ======================================================
# Pedestrian profile
# ======================================================

@dataclass
class PedestrianProfile:
    """Individual physics and behavior parameters for a single pedestrian.

    Each pedestrian gets a unique profile so they walk, react, and
    behave differently — mimicking how different humans have different
    gaits, speeds, and levels of unpredictability.
    """

    base_speed: float       # m/s — comfortable walking speed
    max_speed: float        # m/s — physical sprint limit
    accel_limit: float      # m/s² — max acceleration/deceleration
    damping: float          # how strongly speed returns to base_speed
    heading_noise: float    # baseline heading drift (rad std)
    accel_noise: float      # baseline acceleration noise (m/s² std)

    # Sudden event probabilities (per frame)
    p_sudden_stop: float
    p_direction_flip: float
    p_speed_burst: float
    p_lateral_jerk: float
    p_hesitation: float

    # Event duration ranges (frames)
    stop_duration: tuple = (5, 15)
    burst_duration: tuple = (8, 20)
    jerk_duration: tuple = (3, 8)
    hesitation_duration: tuple = (10, 25)


def generate_profiles(n, rng=None):
    """Generate n diverse pedestrian profiles for LIVE SIMULATION (testing).

    These parameter ranges are intentionally SHIFTED from the training
    model (training_pedestrian.py) to create a distributional mismatch.
    Pedestrians here are slightly faster, jerkier, and more unpredictable
    than what the offline dataset captured — this tests the "robust" in
    Distributionally Robust MPC.
    """

    if rng is None:
        rng = np.random.default_rng()

    profiles = []
    for i in range(n):
        # Assign a rough "type" to get a good mix, then add variation
        ptype = i % 5  # cycles through 5 archetypes

        if ptype == 0:
            # Slow / elderly — slightly faster than training
            base_speed = rng.uniform(0.6, 0.95)
            max_speed = rng.uniform(1.2, 1.6)
            accel_limit = rng.uniform(0.4, 0.85)
            damping = rng.uniform(0.05, 0.09)
            heading_noise = rng.uniform(0.03, 0.06)
            accel_noise = rng.uniform(0.15, 0.28)
            p_sudden_stop = rng.uniform(0.004, 0.010)
            p_direction_flip = rng.uniform(0.002, 0.005)
            p_speed_burst = rng.uniform(0.002, 0.004)
            p_lateral_jerk = rng.uniform(0.004, 0.007)
            p_hesitation = rng.uniform(0.005, 0.010)

        elif ptype == 1:
            # Normal walker — more variable than training
            base_speed = rng.uniform(1.0, 1.5)
            max_speed = rng.uniform(1.8, 2.3)
            accel_limit = rng.uniform(1.0, 1.8)
            damping = rng.uniform(0.03, 0.06)
            heading_noise = rng.uniform(0.03, 0.06)
            accel_noise = rng.uniform(0.28, 0.45)
            p_sudden_stop = rng.uniform(0.005, 0.012)
            p_direction_flip = rng.uniform(0.003, 0.007)
            p_speed_burst = rng.uniform(0.004, 0.008)
            p_lateral_jerk = rng.uniform(0.006, 0.012)
            p_hesitation = rng.uniform(0.004, 0.007)

        elif ptype == 2:
            # Brisk / hurried — faster and more aggressive than training
            base_speed = rng.uniform(1.4, 1.8)
            max_speed = rng.uniform(2.3, 2.8)
            accel_limit = rng.uniform(1.4, 2.2)
            damping = rng.uniform(0.02, 0.05)
            heading_noise = rng.uniform(0.02, 0.04)
            accel_noise = rng.uniform(0.30, 0.50)
            p_sudden_stop = rng.uniform(0.003, 0.008)
            p_direction_flip = rng.uniform(0.002, 0.005)
            p_speed_burst = rng.uniform(0.008, 0.015)
            p_lateral_jerk = rng.uniform(0.005, 0.010)
            p_hesitation = rng.uniform(0.002, 0.005)

        elif ptype == 3:
            # Distracted / erratic — significantly more chaotic than training
            base_speed = rng.uniform(0.8, 1.4)
            max_speed = rng.uniform(1.7, 2.3)
            accel_limit = rng.uniform(0.6, 1.8)
            damping = rng.uniform(0.02, 0.05)
            heading_noise = rng.uniform(0.06, 0.12)
            accel_noise = rng.uniform(0.45, 0.65)
            p_sudden_stop = rng.uniform(0.012, 0.022)
            p_direction_flip = rng.uniform(0.008, 0.015)
            p_speed_burst = rng.uniform(0.005, 0.012)
            p_lateral_jerk = rng.uniform(0.012, 0.022)
            p_hesitation = rng.uniform(0.010, 0.018)

        else:
            # Jogger — faster and more explosive than training
            base_speed = rng.uniform(1.7, 2.3)
            max_speed = rng.uniform(2.8, 3.5)
            accel_limit = rng.uniform(1.8, 3.0)
            damping = rng.uniform(0.03, 0.06)
            heading_noise = rng.uniform(0.02, 0.03)
            accel_noise = rng.uniform(0.20, 0.40)
            p_sudden_stop = rng.uniform(0.002, 0.005)
            p_direction_flip = rng.uniform(0.002, 0.004)
            p_speed_burst = rng.uniform(0.004, 0.008)
            p_lateral_jerk = rng.uniform(0.003, 0.006)
            p_hesitation = rng.uniform(0.002, 0.004)

        profiles.append(PedestrianProfile(
            base_speed=base_speed,
            max_speed=max_speed,
            accel_limit=accel_limit,
            damping=damping,
            heading_noise=heading_noise,
            accel_noise=accel_noise,
            p_sudden_stop=p_sudden_stop,
            p_direction_flip=p_direction_flip,
            p_speed_burst=p_speed_burst,
            p_lateral_jerk=p_lateral_jerk,
            p_hesitation=p_hesitation,
        ))

    return profiles


# ======================================================
# Pedestrian group
# ======================================================

class PedestrianGroup:
    """All pedestrians' state, kept as vectorized NumPy arrays for speed,
    wrapped in a class so it has a clean, ID-based API. No rendering here.

    Physics (speed, acceleration) is defined in real-world units and
    clipped to a fixed physical operating envelope, same principle as
    Vehicle -- see units.py for the real<->sim conversion.

    Each pedestrian has an individual PedestrianProfile that defines
    their unique movement physics (base speed, acceleration limits,
    noise levels) and their probability of sudden behavior events
    (stops, direction flips, speed bursts, lateral jerks, hesitation).

    Movement model: acceleration-driven speed (damped toward each
    pedestrian's own base_speed, plus per-pedestrian random jitter)
    + slowly drifting heading + stochastic sudden events.
    """

    # Absolute physical floor/ceiling (no pedestrian can exceed these
    # regardless of their profile)
    MIN_SPEED_REAL = 0.0       # m/s
    MAX_SPEED_REAL = 3.0       # m/s   (covers joggers)

    # Cooldown after an event ends (frames) -- prevents unrealistic
    # rapid-fire event triggering
    EVENT_COOLDOWN = 10

    def __init__(self, num_pedestrians, road_y, road_width, world_width,
                 seed=None):

        self.n = num_pedestrians
        self.road_y = road_y
        self.road_width = road_width
        self.world_width = world_width
        self.margin = 8  # Extra space outside the road (sidewalks)

        self.rng = np.random.default_rng(seed)
        self.ids = np.arange(self.n)

        # ---- Generate individual profiles ----
        self.profiles = generate_profiles(self.n, self.rng)

        # ---- Vectorize profile parameters for fast NumPy operations ----
        self.base_speed_sim = np.array([
            real_speed_to_sim(p.base_speed) for p in self.profiles
        ])
        self.max_speed_sim = np.array([
            real_speed_to_sim(p.max_speed) for p in self.profiles
        ])
        self.accel_limit_sim = np.array([
            real_accel_to_sim(p.accel_limit) for p in self.profiles
        ])
        self.damping = np.array([p.damping for p in self.profiles])
        self.heading_noise_std = np.array([
            p.heading_noise for p in self.profiles
        ])
        self.accel_noise_std = np.array([
            p.accel_noise for p in self.profiles
        ])

        # Event probabilities (per frame, per pedestrian)
        self.p_sudden_stop = np.array([
            p.p_sudden_stop for p in self.profiles
        ])
        self.p_direction_flip = np.array([
            p.p_direction_flip for p in self.profiles
        ])
        self.p_speed_burst = np.array([
            p.p_speed_burst for p in self.profiles
        ])
        self.p_lateral_jerk = np.array([
            p.p_lateral_jerk for p in self.profiles
        ])
        self.p_hesitation = np.array([
            p.p_hesitation for p in self.profiles
        ])

        # ---- Position & motion state ----
        self.x = self.rng.uniform(20, 95, size=self.n)
        self.y = self.rng.uniform(
            road_y - road_width / 2 - self.margin,
            road_y + road_width / 2 + self.margin,
            size=self.n
        )
        self.heading = self.rng.uniform(0, 2 * np.pi, size=self.n)
        self.speed_sim = self.base_speed_sim.copy()

        self.last_accel_real = np.zeros(self.n)  # for display / logging

        # ---- Event state ----
        self.active_event = np.full(self.n, EVENT_NONE, dtype=int)
        self.event_timer = np.zeros(self.n, dtype=int)
        self.cooldown_timer = np.zeros(self.n, dtype=int)

        # Hesitation uses an internal toggle (move vs. pause)
        self.hesitation_phase = np.zeros(self.n, dtype=int)

    @property
    def speed_real(self):
        """Current speed of every pedestrian in real-world units (m/s)."""
        return sim_speed_to_real(self.speed_sim)

    # ------------------------------------------------------------------
    # Event triggering
    # ------------------------------------------------------------------

    def _try_trigger_events(self):
        """For pedestrians not currently in an event or cooldown, roll
        dice to see if a sudden behavior event triggers this frame.

        Priority order (first match wins):
        sudden_stop > direction_flip > speed_burst > lateral_jerk > hesitation
        """

        eligible = (
            (self.active_event == EVENT_NONE) & (self.cooldown_timer <= 0)
        )

        if not np.any(eligible):
            return

        rolls = self.rng.uniform(0, 1, size=(5, self.n))

        # Check each event type in priority order
        triggered = np.zeros(self.n, dtype=bool)

        # Sudden stop
        mask = eligible & ~triggered & (rolls[0] < self.p_sudden_stop)
        if np.any(mask):
            indices = np.where(mask)[0]
            self.active_event[indices] = EVENT_SUDDEN_STOP
            for idx in indices:
                dur = self.profiles[idx].stop_duration
                self.event_timer[idx] = self.rng.integers(dur[0], dur[1] + 1)
            triggered |= mask

        # Direction flip
        mask = eligible & ~triggered & (rolls[1] < self.p_direction_flip)
        if np.any(mask):
            indices = np.where(mask)[0]
            self.active_event[indices] = EVENT_DIRECTION_FLIP
            self.event_timer[indices] = 1  # instant, single-frame trigger
            triggered |= mask

        # Speed burst
        mask = eligible & ~triggered & (rolls[2] < self.p_speed_burst)
        if np.any(mask):
            indices = np.where(mask)[0]
            self.active_event[indices] = EVENT_SPEED_BURST
            for idx in indices:
                dur = self.profiles[idx].burst_duration
                self.event_timer[idx] = self.rng.integers(dur[0], dur[1] + 1)
            triggered |= mask

        # Lateral jerk
        mask = eligible & ~triggered & (rolls[3] < self.p_lateral_jerk)
        if np.any(mask):
            indices = np.where(mask)[0]
            self.active_event[indices] = EVENT_LATERAL_JERK
            for idx in indices:
                dur = self.profiles[idx].jerk_duration
                self.event_timer[idx] = self.rng.integers(dur[0], dur[1] + 1)
            # Apply the sharp heading change immediately
            jerk_dir = self.rng.choice([-1, 1], size=len(indices))
            self.heading[indices] += jerk_dir * (
                np.pi / 2 + self.rng.normal(0, 0.1, size=len(indices))
            )
            triggered |= mask

        # Hesitation
        mask = eligible & ~triggered & (rolls[4] < self.p_hesitation)
        if np.any(mask):
            indices = np.where(mask)[0]
            self.active_event[indices] = EVENT_HESITATION
            for idx in indices:
                dur = self.profiles[idx].hesitation_duration
                self.event_timer[idx] = self.rng.integers(dur[0], dur[1] + 1)
            self.hesitation_phase[indices] = 0
            triggered |= mask

    # ------------------------------------------------------------------
    # Movement step
    # ------------------------------------------------------------------

    def step(self):
        """Advance all pedestrians by one timestep (vectorized)."""

        # ---- Phase 1: Check for new events ----
        self._try_trigger_events()

        # ---- Phase 2: Movement ----
        min_speed_sim = real_speed_to_sim(self.MIN_SPEED_REAL)

        # --- Normal movement (pedestrians with no active event) ---
        normal = self.active_event == EVENT_NONE

        if np.any(normal):
            noise_accel_real = self.rng.normal(
                0, self.accel_noise_std[normal]
            )
            accel_sim = (
                self.damping[normal]
                * (self.base_speed_sim[normal] - self.speed_sim[normal])
                + real_accel_to_sim(noise_accel_real)
            )
            # Clip acceleration to per-pedestrian limits
            accel_sim = np.clip(
                accel_sim,
                -self.accel_limit_sim[normal],
                self.accel_limit_sim[normal]
            )
            self.speed_sim[normal] = np.clip(
                self.speed_sim[normal] + accel_sim,
                min_speed_sim,
                self.max_speed_sim[normal]
            )
            self.heading[normal] += self.rng.normal(
                0, self.heading_noise_std[normal]
            )

        # --- Sudden stop: speed -> 0 ---
        stopped = self.active_event == EVENT_SUDDEN_STOP
        if np.any(stopped):
            self.speed_sim[stopped] = 0.0

        # --- Direction flip: reverse heading (single-frame trigger) ---
        flipping = self.active_event == EVENT_DIRECTION_FLIP
        if np.any(flipping):
            flip_noise = self.rng.normal(0, 0.15, size=int(np.sum(flipping)))
            self.heading[flipping] += np.pi + flip_noise

        # --- Speed burst: push toward max speed ---
        bursting = self.active_event == EVENT_SPEED_BURST
        if np.any(bursting):
            burst_accel = 0.5 * (
                self.max_speed_sim[bursting] - self.speed_sim[bursting]
            )
            self.speed_sim[bursting] = np.clip(
                self.speed_sim[bursting] + burst_accel,
                min_speed_sim,
                self.max_speed_sim[bursting]
            )
            # Hold heading relatively steady during burst
            self.heading[bursting] += self.rng.normal(
                0, 0.005, size=int(np.sum(bursting))
            )

        # --- Lateral jerk: heading was already shifted on trigger,
        #     just maintain slightly elevated speed ---
        jerking = self.active_event == EVENT_LATERAL_JERK
        if np.any(jerking):
            bump = real_accel_to_sim(0.5)
            self.speed_sim[jerking] = np.clip(
                self.speed_sim[jerking] + bump,
                min_speed_sim,
                self.max_speed_sim[jerking]
            )

        # --- Hesitation: alternate between moving and pausing ---
        hesitating = self.active_event == EVENT_HESITATION
        if np.any(hesitating):
            # Toggle every 3-4 frames
            toggle_mask = hesitating & (self.event_timer % 4 == 0)
            self.hesitation_phase[toggle_mask] = (
                1 - self.hesitation_phase[toggle_mask]
            )
            # Phase 0 = pause, phase 1 = move
            paused = hesitating & (self.hesitation_phase == 0)
            moving = hesitating & (self.hesitation_phase == 1)
            self.speed_sim[paused] *= 0.3  # slow to near-stop
            # Moving phase: push back toward base speed
            if np.any(moving):
                self.speed_sim[moving] += 0.3 * (
                    self.base_speed_sim[moving] - self.speed_sim[moving]
                )

        # ---- Apply position update for ALL pedestrians ----
        self.x += self.speed_sim * np.cos(self.heading)
        self.y += self.speed_sim * np.sin(self.heading)

        # ---- Phase 3: Boundary handling ----
        # Clamp position first, then flip heading

        # X boundaries
        out_left = self.x < 0
        out_right = self.x > self.world_width
        self.x = np.clip(self.x, 0, self.world_width)
        self.heading = np.where(
            out_left | out_right, np.pi - self.heading, self.heading
        )

        # Y boundaries (road edges + sidewalks)
        y_min = self.road_y - self.road_width / 2 - self.margin
        y_max = self.road_y + self.road_width / 2 + self.margin
        out_bottom = self.y < y_min
        out_top = self.y > y_max
        self.y = np.clip(self.y, y_min, y_max)
        self.heading = np.where(
            out_bottom | out_top, -self.heading, self.heading
        )

        # ---- Phase 4: Decrement event timers ----
        active = self.active_event != EVENT_NONE
        self.event_timer[active] -= 1

        # Events that have expired
        expired = active & (self.event_timer <= 0)
        self.active_event[expired] = EVENT_NONE
        self.cooldown_timer[expired] = self.EVENT_COOLDOWN

        # Decrement cooldown for non-active pedestrians
        cooling = (~active) & (self.cooldown_timer > 0)
        self.cooldown_timer[cooling] -= 1

        # Store last accel for display (approximate from speed change)
        # This is a simplification; detailed per-event accel tracking
        # could be added later if needed for the controller.
        self.last_accel_real = np.zeros(self.n)

    @property
    def positions(self):
        """(N, 2) array of [x, y] for every pedestrian."""
        return np.column_stack([self.x, self.y])
