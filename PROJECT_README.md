# DR-MPC Pedestrian-Aware Longitudinal Speed Control

## What this project is

This project builds a self-driving car's speed controller that has to
share the road with unpredictable pedestrians. The car can only speed
up or slow down (no steering) — the goal is to keep a safe distance
from pedestrians crossing its path while still driving efficiently,
even when pedestrian behavior doesn't match what the controller
expects.

The core idea is **Distributionally Robust Model Predictive Control
(DR-MPC)**: instead of assuming pedestrians move in one fixed,
predictable way, the controller plans against a *range* of plausible
pedestrian behaviors at once, and optimizes for the worst case within
that range. This makes it robust to real-world pedestrians behaving
differently than any single assumed model.

## The overall approach

1. **Build a simulation environment** — a simple 2D world with a road,
   an ego vehicle, and pedestrians, so the whole pipeline can be
   developed and tested without needing real sensor data.

2. **Collect historical pedestrian data** — run many independent,
   short pedestrian trajectories and record them. This stands in for
   what a real system would collect from logged sensor data over time.

3. **Extract behavioral residuals** — for each historical trajectory,
   compare what actually happened to what simple physics alone would
   have predicted (a pedestrian just continuing in a straight line).
   The *difference* is the part physics can't explain — hesitation,
   speeding up, sudden turns, etc. This residual pool is what captures
   "how unpredictable real pedestrians tend to be."

4. **Characterize uncertainty** — use the residual pool to build a
   statistical notion of "distance" between different possible
   pedestrian behaviors, and construct an ambiguity set (a
   Wasserstein ball) around the expected behavior. This is the
   mathematical object the controller optimizes against — described
   in full in the separate mathematical formulation, deliberately not
   included here.

5. **Anchor to live sensor data** — at every control step, take the
   vehicle's current (simulated) sensor reading of the pedestrian and
   combine it with the historical residual pool to generate a live,
   up-to-date set of plausible future pedestrian trajectories.

6. **Solve the robust control problem** — every control step, solve an
   optimization that picks the safest and smoothest acceleration,
   accounting for the worst case among the plausible pedestrian
   trajectories, subject to a hard safety-distance floor as a
   failsafe.

7. **Stress-test with unseen behavior** — the pedestrians used to test
   the finished controller are deliberately built with different,
   more erratic movement patterns (sudden darting, stop-and-go,
   zigzagging, random jerks, bouncing around the world) than the
   pedestrians used to build the historical dataset in step 2. This is
   intentional: it's what actually tests whether the controller
   generalizes, rather than just replaying the same behavior it was
   built from.

## Project structure (high level)

- **Simulation environment** — the vehicle, pedestrians, and a simple
  LiDAR-style sensor, plus rendering to visualize runs as GIFs.
- **Historical data & residual modules** — generate and process the
  offline pedestrian trajectory dataset described in steps 2–3 above.
- **Uncertainty / ambiguity set modules** — turn the residual dataset
  into the statistical object the controller reasons about (step 4).
- **Controller** — the actual robust optimization solved at every
  control step (steps 5–6).
- **Test scenario scripts** — build the stress-testing pedestrians
  (step 7) and run/record full simulated drives for review.

## Status

The environment, historical data pipeline, ambiguity-set
construction, and the robust controller are all implemented and
wired together end to end. Current work is focused on stress-testing
the controller against increasingly unpredictable pedestrian behavior
to find and understand its limits before refining the approach
further.
