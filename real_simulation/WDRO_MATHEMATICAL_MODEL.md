# Distributionally Robust Model Predictive Control (WDRO-MPC)
## Mathematical Formulation and Optimization Specification

---

### 1. Overview and Problem Description

This document details the mathematical model of the longitudinal speed controller for an autonomous vehicle navigating among unpredictable pedestrians. 

The vehicle operates strictly along the road centerline with one-dimensional longitudinal control (acceleration/braking). The controller optimizes vehicle acceleration over a receding lookahead horizon of N steps, guaranteeing collision avoidance against the worst-case probability distribution of pedestrian motions within a Wasserstein ambiguity ball.

---

### 2. Nomenclature and System Dimensions

| Symbol | Description | Standard Value | Units |
| :--- | :--- | :--- | :--- |
| N | Prediction horizon steps | 20 | steps |
| Δt | Discrete time step | 0.2 | seconds (5 Hz) |
| T | Total lookahead duration (N × Δt) | 4.0 | seconds |
| S | Number of historical scenario residuals | 100 | scenarios |
| D | Number of detected pedestrians within LiDAR | Dynamic (0 to 5) | pedestrians |
| R_lidar | LiDAR circular sensing radius | 12.0 | meters |
| v_des | Desired cruising speed | 8.0 | m/s (~28.8 km/h) |
| g_min | Minimum required longitudinal safety distance | 5.0 | meters |
| ε | Wasserstein ambiguity ball radius | 0.1 | unitless |
| a_min, a_max | Minimum braking and maximum acceleration limits | -5.0, +3.0 | m/s² |
| v_min, v_max | Minimum and maximum admissible speeds | 0.0, 15.0 | m/s |
| w_v | Speed tracking error weight | 1.0 | unitless |
| w_a | Acceleration comfort / control effort weight | 0.5 | unitless |
| w_slack | Soft-slack safety violation penalty weight | 1000.0 | unitless |

---

### 3. Vehicle Kinematic State and Dynamics

The vehicle state at discrete time step k along the horizon is defined by:
* Longitudinal position: x[k]
* Longitudinal velocity: v[k]
* Longitudinal control acceleration: a[k]

#### Initial State Conditions (at step k = 0):
```
x[0] = x_current
v[0] = v_current
```

#### State Transition Equations (for k = 0, 1, ..., N - 1):
```
x[k+1] = x[k]  +  v[k] · Δt  +  ½ · a[k] · (Δt)²
v[k+1] = v[k]  +  a[k] · Δt
```

#### Physical Kinematic Bounds:
For every step k = 0, 1, ..., N - 1:
```
a_min ≤ a[k] ≤ a_max       (-5.0 ≤ a[k] ≤ +3.0 m/s²)
v_min ≤ v[k] ≤ v_max       ( 0.0 ≤ v[k] ≤ 15.0 m/s)
```

---

### 4. Pedestrian Uncertainty and Scenario Representation

For each pedestrian d detected within the 12.0 m LiDAR radius (where d = 1, ..., D):
* The system evaluates S = 100 plausible future trajectories over the horizon k = 1, ..., N.
* The predicted longitudinal position of pedestrian d under historical scenario s at step k is denoted:
  p_x[d, s, k]

#### Nominal (Empirical Mean) Pedestrian Trajectory:
For each step k = 1, ..., N:
```
p̄_x[d, k] = (1 / S) · ∑_{s=1}^{S} p_x[d, s, k]
```

#### Ground Transport Cost Matrix:
* M is a precomputed (S × S) symmetric matrix of pairwise Wasserstein transportation distances between empirical residual scenarios:
  M[s, s'] ≥ 0,  with M[s, s] = 0

---

### 5. Safety Constraints and Loss Functions

#### A. Nominal Safety Floor (with Soft-Slack)
To guarantee solver feasibility during tight/emergency situations while heavily penalizing unsafe margins, a non-negative slack variable s[k] is introduced:
For each detected pedestrian d = 1, ..., D and each step k = 0, ..., N - 1:
```
p̄_x[d, k+1]  -  x[k+1]  +  s[k]  ≥  g_min

with:
  s[k] ≥ 0
```

#### B. Individual Scenario Violation Loss
For each pedestrian d, each scenario s = 1, ..., S, and each horizon step k = 1, ..., N:
```
Longitudinal Gap:
  gap[d, s, k] = p_x[d, s, k] - x[k]

Safety Violation:
  violation[d, s, k] = max( 0,  g_min - gap[d, s, k] )
```

#### C. Cumulative Scenario Loss Auxiliary Variable:
For each scenario s = 1, ..., S:
```
h[d, s]  ≥  ∑_{k=1}^{N} (violation[d, s, k])²

with:
  h[d, s] ≥ 0
```

---

### 6. Wasserstein Distributionally Robust Reformulation (Dual Form)

Instead of assuming uniform probability across scenarios, the controller protects against the worst-case probability distribution inside the Wasserstein ball of radius ε around the empirical distribution.

By strong duality of Wasserstein distributionally robust optimization (Kantorovich duality), the worst-case risk for each detected pedestrian d is expressed via dual variables:
* λ[d] ≥ 0 (dual multiplier for Wasserstein ball radius)
* u[d, s] ≥ 0 (epigraph variables for each scenario s = 1, ..., S)

#### Dual Epigraph Constraint:
For every scenario pair (s, s') where s, s' ∈ {1, ..., S}:
```
u[d, s]  ≥  h[d, s']  -  λ[d] · M[s, s']
```

In matrix form across all 100 scenarios:
```
Vector u[d] (column S × 1)  ≥  Row Vector h[d] (1 × S)  -  λ[d] · Matrix M (S × S)
```

---

### 7. Complete Optimization Problem (Objective Function)

The controller solves the following convex program at every control step:

```
Minimize J = J_tracking  +  J_comfort  +  J_slack  +  J_WDRO
```

Subject to:
1. Vehicle initial conditions and kinematic transitions (Section 3)
2. Actuator and speed bounds (Section 3)
3. Nominal soft-slack safety floor (Section 5A)
4. Scenario loss bounds (Section 5C)
5. Dual Wasserstein epigraph constraints (Section 6)
6. Non-negativity: λ[d] ≥ 0,  u[d, s] ≥ 0,  h[d, s] ≥ 0,  s[k] ≥ 0

#### Cost Function Breakdown:

```
1. Velocity Tracking Cost:
   J_tracking = ∑_{k=0}^{N-1}  w_v · (v[k] - v_des)²

2. Control Effort / Passenger Comfort Cost:
   J_comfort  = ∑_{k=0}^{N-1}  w_a · (a[k])²

3. Slack Penalty (Emergency Safety Buffer):
   J_slack    = ∑_{k=0}^{N-1}  w_slack · (s[k])²

4. Worst-Case Distributional Risk Penalty:
   If D > 0 (pedestrians detected):
     J_WDRO   = ∑_{d=1}^{D} [ λ[d] · ε  +  (1 / S) · ∑_{s=1}^{S} u[d, s] ]
   If D = 0 (road clear):
     J_WDRO   = 0
```

---

### 8. Receding Horizon Control Law

At each execution step:
1. The solver computes the optimal acceleration sequence:
   a* = [ a*[0], a*[1], ..., a*[N-1] ]
2. Only the first control decision is applied to the vehicle:
   a_cmd = a*[0]
3. The horizon advances by one step Δt = 0.2 s, state measurements are refreshed, and the optimization is repeated.
