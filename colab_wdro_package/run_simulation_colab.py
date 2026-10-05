"""Headless Closed-Loop WDRO-MPC Simulation Runner for Google Colab.

Runs closed-loop autonomous speed regulation against multi-pedestrian traffic.
Supports BOTH solvers:
  - Standard Frozen Reference Model (Convex QP)
  - Circular Model (Sequential Convex Programming)

Generates:
  1. An HTML5 / MP4 animated video of the simulation (actors, LiDAR, prediction fans).
  2. A 4-panel publication telemetry plot (Speed, Acceleration, Distance Gap, Latency).

Usage:
  python run_simulation_colab.py --model frozen    # Standard Model
  python run_simulation_colab.py --model circular  # Circular SCP Model
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

from environment import Environment
from units import DT
from threat_assessment import predict_detected_pedestrians
from wdro_mpc import solve_wdro_mpc
from wdro_mpc_circular import solve_wdro_mpc_circular


def run_colab_simulation(model="frozen", num_cycles=2, save_video=True, save_plots=True):
    print("=" * 72)
    print(f"  STARTING CLOSED-LOOP WDRO-MPC SIMULATION (MODEL: {model.upper()})")
    print("=" * 72)

    # 1. Load Precomputed Behavioral Artifacts
    residuals = np.load("residuals.npy")[:, 1:21, :2]
    dist_matrix = np.load("distance_matrix.npy")
    S, N, _ = residuals.shape

    # For faster interactive animation in Colab when testing circular, use S_live = 25 or full 100
    if model == "circular":
        S_live = 25
        res_active = residuals[:S_live]
        dist_active = dist_matrix[:S_live, :S_live]
    else:
        S_live = S
        res_active = residuals
        dist_active = dist_matrix

    # 2. Setup Environment
    env = Environment(num_pedestrians=5, car_length=0.5)

    cycle_distance = env.world_width + env.vehicle.length
    frames_per_cycle = int(np.ceil(cycle_distance / env.vehicle.speed_sim))
    total_frames = frames_per_cycle * num_cycles

    print(f"Total Simulation Steps : {total_frames} frames ({total_frames * DT:.1f}s)")
    print(f"Target Cruising Speed  : 8.0 m/s (~28.8 km/h)")
    print(f"Active Scenario Pool   : {S_live} scenarios")

    # Telemetry logging arrays
    time_log = []
    speed_log = []
    accel_log = []
    min_dist_log = []
    solve_time_log = []

    # Frame state history for animation rendering
    history_car = []
    history_peds = []
    history_scenarios = []

    current_accel = 0.0

    # 3. Closed-Loop Execution Loop
    for frame_i in range(total_frames):
        t_now = frame_i * DT
        v = env.vehicle
        detections = env.detections

        current_scenarios = None
        current_detected = None
        t_solve = 0.0

        if detections is not None and len(detections["indices"]) > 0:
            speed_real = env.pedestrians.speed_real
            heading = env.pedestrians.heading
            velocities = np.column_stack([
                speed_real * np.cos(heading),
                speed_real * np.sin(heading)
            ])

            result = predict_detected_pedestrians(
                vehicle_center=v.center,
                detected_indices=detections["indices"],
                pedestrian_positions=env.pedestrians.positions,
                pedestrian_velocities=velocities,
                residuals=res_active,
                horizon=N,
            )

            current_scenarios = result["scenarios"]
            current_detected = result["detected_indices"]

            # Solve optimization program
            t_start = time.perf_counter()
            if model == "circular":
                a_opt, info = solve_wdro_mpc_circular(
                    x0=v.center[0],
                    v0=v.speed_real,
                    scenarios=current_scenarios,
                    dist_matrix=dist_active,
                    v_des=8.0,
                    v_max=15.0,
                    c=15.0,
                    w=0.2,
                    epsilon=0.4,
                    dt=DT,
                    N=N,
                    max_iter=2,
                    conv_tol=0.08,
                )
            else:
                a_opt, info = solve_wdro_mpc(
                    x0=v.center[0],
                    v0=v.speed_real,
                    scenarios=current_scenarios,
                    dist_matrix=dist_active,
                    v_des=8.0,
                    v_max=15.0,
                    c=15.0,
                    w=0.2,
                    epsilon=0.4,
                    dt=DT,
                    N=N,
                )
            t_solve = (time.perf_counter() - t_start) * 1000.0
            current_accel = float(a_opt) if a_opt is not None else -5.0
        else:
            speed_error = 8.0 - v.speed_real
            current_accel = float(np.clip(1.5 * speed_error, -3.0, 2.0))
            t_solve = 0.0

        # Step physics
        env.step(accel_real=current_accel)

        # Log state
        v = env.vehicle
        time_log.append(t_now)
        speed_log.append(v.speed_real)
        accel_log.append(current_accel)
        solve_time_log.append(t_solve)

        # Distance to closest pedestrian
        dx = env.pedestrians.x - v.center[0]
        dy = env.pedestrians.y - v.center[1]
        dist_all = np.sqrt(dx**2 + dy**2)
        min_dist_log.append(np.min(dist_all))

        # Record actors for animation
        history_car.append((v.x, v.y, v.length, v.width, v.center))
        history_peds.append((env.pedestrians.positions.copy(), detections["mask"].copy() if detections is not None else [False]*5))
        history_scenarios.append((current_scenarios, current_detected, env.pedestrians.positions.copy()))

        if frame_i % 20 == 0 or frame_i == total_frames - 1:
            print(f"  Frame {frame_i:03d}/{total_frames} | Speed: {v.speed_real:.1f} m/s | Accel: {current_accel:+.2f} m/s^2 | Solve: {t_solve:.0f} ms | Min Gap: {min_dist_log[-1]:.1f}m")

    # 4. Generate Publication Telemetry Figure
    if save_plots:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        t_arr = np.array(time_log)

        # Panel 1: Speed
        axes[0].plot(t_arr, speed_log, color="#2563eb", linewidth=2, label="Vehicle Speed (m/s)")
        axes[0].axhline(8.0, color="#16a34a", linestyle="--", linewidth=1.5, label="Target Cruise (8.0 m/s)")
        axes[0].set_ylabel("Speed (m/s)", fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"WDRO-MPC Closed-Loop Response [{model.upper()} MODEL]", fontweight="bold", fontsize=14)

        # Panel 2: Acceleration
        axes[1].plot(t_arr, accel_log, color="#dc2626", linewidth=2, label="Commanded Acceleration (m/s^2)")
        axes[1].axhline(0.0, color="black", linestyle="-", linewidth=0.8)
        axes[1].set_ylabel("Accel (m/s^2)", fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right")

        # Panel 3: Minimum Clearance Distance
        axes[2].plot(t_arr, min_dist_log, color="#059669", linewidth=2, label="Distance to Closest Pedestrian (m)")
        axes[2].axhline(12.0, color="#00e5ff", linestyle="--", linewidth=1.2, label="12m LiDAR Perception Limit")
        axes[2].set_ylabel("Clearance (m)", fontweight="bold")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(loc="upper right")

        # Panel 4: Solve Latency
        axes[3].bar(t_arr, solve_time_log, width=0.18, color="#7c3aed", alpha=0.8, label="Solve Latency (ms)")
        axes[3].axhline(200.0, color="#e11d48", linestyle=":", linewidth=1.5, label="200ms Real-Time Deadline")
        axes[3].set_ylabel("Latency (ms)", fontweight="bold")
        axes[3].set_xlabel("Simulation Time (seconds)", fontweight="bold")
        axes[3].grid(True, linestyle=":", alpha=0.6)
        axes[3].legend(loc="upper right")

        plt.tight_layout()
        plot_filename = f"wdro_simulation_telemetry_{model}.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)
        print(f"\n[Telemetry Plot Saved] -> {plot_filename}")

    # 5. Generate Matplotlib Animation Video
    if save_video:
        fig_anim, ax = plt.subplots(figsize=(14, 4), dpi=100)
        ax.set_xlim(0, env.world_width)
        ax.set_ylim(0, env.world_height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#e2efe0")

        # Static road
        road = Rectangle((0, env.road_y - env.road_width / 2), env.world_width, env.road_width, color="#2c343a")
        ax.add_patch(road)
        ax.plot([0, env.world_width], [env.road_y, env.road_y], "--", color="white", linewidth=2)

        # Dynamic actors
        car_patch = Rectangle((0, 0), 0.5, 2.0, color="#1d4ed8", zorder=5)
        ax.add_patch(car_patch)
        lidar_patch = Circle((0, 0), env.lidar.radius, fill=False, linestyle="--", edgecolor="#00e5ff", linewidth=1.8, zorder=3)
        ax.add_patch(lidar_patch)

        ped_scatter = ax.scatter([], [], s=80, zorder=6)
        traj_collection = LineCollection([], colors="#ff6d00", linewidths=1.2, alpha=0.5, zorder=4)
        ax.add_collection(traj_collection)

        hud_text = ax.text(10, env.world_height - 3, "", fontsize=10, fontweight="bold", color="#0f172a",
                            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1", lw=1))

        def update(frame):
            cx, cy, clen, cwid, center = history_car[frame]
            car_patch.set_xy((cx, cy - cwid / 2.0))
            car_patch.set_width(clen)
            car_patch.set_height(cwid)
            lidar_patch.center = center

            ped_pos, ped_mask = history_peds[frame]
            ped_scatter.set_offsets(ped_pos)
            colors = ["#00e676" if m else "#ef4444" for m in ped_mask]
            ped_scatter.set_facecolors(colors)
            ped_scatter.set_edgecolors("#0f172a")

            scen, det, ppos = history_scenarios[frame]
            segments = []
            if scen is not None and det is not None:
                for d in range(len(det)):
                    p0 = ppos[det[d]]
                    for s in range(0, scen.shape[1], max(1, scen.shape[1] // 15)):
                        traj = scen[d, s]
                        segments.append(np.vstack([p0[None, :], traj]))
            traj_collection.set_segments(segments)

            hud_text.set_text(
                f"Model: {model.upper()} | Time: {frame * DT:.1f}s | Speed: {speed_log[frame]:.1f} m/s | Accel: {accel_log[frame]:+.2f} m/s^2 | Solve: {solve_time_log[frame]:.0f}ms"
            )
            return car_patch, lidar_patch, ped_scatter, traj_collection, hud_text

        anim = animation.FuncAnimation(fig_anim, update, frames=total_frames, interval=200, blit=True)

        video_filename = f"wdro_simulation_animation_{model}.mp4"
        try:
            anim.save(video_filename, writer="ffmpeg", fps=5, dpi=100)
            print(f"[Simulation Video Saved] -> {video_filename}")
        except Exception:
            # Fallback to HTML animation if ffmpeg is missing
            html_filename = f"wdro_simulation_animation_{model}.html"
            with open(html_filename, "w", encoding="utf-8") as f:
                f.write(anim.to_jshtml())
            print(f"[HTML5 Video Saved] -> {html_filename}")
        plt.close(fig_anim)

    print("\n" + "=" * 72)
    print(f"  SIMULATION COMPLETED SUCCESSFULLY ({model.upper()} MODEL)")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Google Colab WDRO-MPC Simulation")
    parser.add_argument("--model", type=str, choices=["frozen", "circular"], default="frozen",
                        help="Choose solver formulation: 'frozen' or 'circular'")
    parser.add_argument("--cycles", type=int, default=2, help="Number of road lap cycles (default: 2)")
    args = parser.parse_args()

    run_colab_simulation(model=args.model, num_cycles=args.cycles)
