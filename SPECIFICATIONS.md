# Project Specifications and Standards

## What this document is

This document outlines the standard configuration values, parameters,
and physical metrics used across the project to maintain consistency
across the simulation environment, data collection pipeline, and robust
controller modules.

## Time and sampling specifications

- **Simulation time step (Δt / DT = 0.2 s)** — Physical duration of each
  simulation and prediction step (5 Hz update rate). Fixed standard
  across all modules.

- **Prediction horizon (N = 20 steps)** — Number of future steps
  predicted by the nominal model and the MPC solver.

- **Horizon duration (T_horizon = 4.0 s)** — Total lookahead time span,
  derived as N × Δt.

- **Historical sample duration (T_sample = 4.0 s)** — Duration of each
  recorded offline trajectory (21 discrete timesteps from t = 0 to
  t = 20).

- **Historical sample size (S = 100 samples)** — Total number of
  historical pedestrian trajectory samples collected to construct the
  offline dataset and residual pool.

- **Control update interval (CONTROL_EVERY = 1 or 5 steps)** — Receding
  horizon solver execution frequency.

## Pedestrian population specifications

- **Training phase (10 pedestrians)** — 10 pedestrians simulated
  simultaneously in the environment to collect trajectory batches until
  the S = 100 sample threshold is reached.

- **Testing / real-time phase (5 pedestrians)** — 5 active pedestrians
  in the live environment for real-time threat assessment and vehicle
  control.

- **Live LiDAR detections (D ≤ 5 pedestrians)** — Dynamic count of
  pedestrians detected within the 12.0 m sensor radius at any given
  control step.

## Environment and road specifications

- **World dimensions** — Continuous 2D simulation domain spanning 120.0 m
  in length and 40.0 m in width, with periodic boundary wrapping along the
  x-axis.

- **Roadway layout** — Centerline located at y = 20.0 m with a total
  width of 12.0 m (road region spans y in [14.0, 26.0] m).

- **Sidewalks** — Pedestrian zones located on both sides of the roadway
  (lower sidewalk in [0.0, 14.0] m; upper sidewalk in [26.0, 40.0] m).

## Vehicle and sensor specifications

- **Vehicle dimensions** — Compact ego vehicle with length of 3.0 m and
  width of 2.0 m.

- **Desired cruising speed (v_des = 8.0 m/s)** — Target longitudinal
  cruising speed (~28.8 km/h) under unconstrained conditions.

- **Speed limits [vmin, vmax = 0.0, 15.0 m/s]** — Admissible longitudinal
  speed bounds (0.0 to ~54.0 km/h).

- **Acceleration limits [amin, amax = -5.0, +3.0 m/s²]** — Maximum
  braking deceleration of -5.0 m/s² and maximum acceleration of +3.0 m/s².

- **LiDAR detection radius (R = 12.0 m)** — Circular range sensing zone
  centered at the vehicle geometric center.
