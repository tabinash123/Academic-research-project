import pygame
import numpy as np


class PygamePredictionRenderer:
    """Hardware-accelerated real-time visualizer for DR-MPC scenario prediction.

    Renders environment state, vehicle LiDAR sensing, and live future trajectory
    candidate fans (anchored by historical residuals) with high visual clarity,
    transparent LiDAR overlay, and near-zero CPU/RAM cost.
    """

    def __init__(self, env, scale=10, title="Real-Time DR-MPC Pedestrian Scenario Prediction"):
        self.env = env
        self.scale = scale  # 10 pixels per meter (120m -> 1200px, 40m -> 400px)

        self.screen_width = int(env.world_width * self.scale)
        self.screen_height = int(env.world_height * self.scale)

        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(title)

        # High-contrast modern research color palette
        self.COLOR_GRASS = (226, 239, 224)          # Soft muted grass sidewalk (#e2efe0)
        self.COLOR_ROAD = (44, 52, 58)              # Deep dark asphalt (#2c343a)
        self.COLOR_CURB = (180, 190, 185)           # Concrete road boundary
        self.COLOR_CENTERLINE = (255, 255, 255)     # Crisp white dashed centerline
        
        # Vehicle colors
        self.COLOR_CAR = (29, 78, 216)              # Deep royal blue (#1d4ed8)
        self.COLOR_CAR_ROOF = (59, 130, 246)        # Lighter blue roof highlight
        self.COLOR_CAR_OUTLINE = (15, 23, 42)       # Dark border
        self.COLOR_HEADLIGHT = (254, 240, 138)      # Pale yellow front indicator

        # LiDAR sensing (Clean outline ring - NO obstructing solid wash)
        self.COLOR_LIDAR_RING = (0, 229, 255)       # Electric cyan (#00e5ff)
        self.COLOR_LIDAR_AURA = (0, 229, 255, 12)   # Ultra-sheer 5% atmospheric wash

        # Pedestrian styling
        self.COLOR_PED_NORMAL = (239, 68, 68)       # Coral red (#ef4444)
        self.COLOR_PED_DETECTED = (0, 230, 118)     # Bright neon emerald (#00e676)
        self.COLOR_PED_OUTLINE = (15, 23, 42)       # Dark border
        self.COLOR_PED_DETECTED_RING = (255, 255, 255) # Inner white ring for maximum pop

        # Scenario trajectory lines
        self.COLOR_TRAJ_LINE = (255, 109, 0, 50)    # Vivid Amber/Safety Orange (#ff6d00) with alpha

        # UI & Telemetry
        self.COLOR_TEXT = (15, 23, 42)
        self.font_hud = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_labels = pygame.font.SysFont("Arial", 10, bold=True)
        self.font_plan = pygame.font.SysFont("Arial", 11, bold=True)

        # Dedicated transparent surface for alpha-blended trajectory lines and subtle aura
        self.alpha_surface = pygame.Surface(
            (self.screen_width, self.screen_height),
            pygame.SRCALPHA
        )

    def to_screen_x(self, x_real):
        """Convert world X (meters) to screen X (pixels)."""
        return int(x_real * self.scale)

    def to_screen_y(self, y_real):
        """Convert world Y (meters, origin bottom) to screen Y (pixels, origin top)."""
        return int((self.env.world_height - y_real) * self.scale)

    def to_screen_coords(self, xy_array):
        """Vectorized conversion of (N, 2) world coordinates to (N, 2) screen pixels."""
        x_px = (xy_array[:, 0] * self.scale).astype(np.int32)
        y_px = ((self.env.world_height - xy_array[:, 1]) * self.scale).astype(np.int32)
        return np.column_stack([x_px, y_px])

    def draw_static_world(self):
        """Draw grass sidewalks, asphalt road surface, curbs, and dashed centerline."""
        # 1. Background sidewalks
        self.screen.fill(self.COLOR_GRASS)

        # 2. Road surface
        road_top_real = self.env.road_y + self.env.road_width / 2.0
        road_screen_y = self.to_screen_y(road_top_real)
        road_screen_height = int(self.env.road_width * self.scale)

        pygame.draw.rect(
            self.screen,
            self.COLOR_ROAD,
            (0, road_screen_y, self.screen_width, road_screen_height)
        )

        # 3. Curbs (edge lines)
        pygame.draw.line(self.screen, self.COLOR_CURB, (0, road_screen_y), (self.screen_width, road_screen_y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURB, (0, road_screen_y + road_screen_height), (self.screen_width, road_screen_y + road_screen_height), 2)

        # 4. Dashed white centerline
        centerline_y = self.to_screen_y(self.env.road_y)
        dash_length = int(3.0 * self.scale)
        gap_length = int(2.0 * self.scale)
        x = 0
        while x < self.screen_width:
            pygame.draw.line(
                self.screen,
                self.COLOR_CENTERLINE,
                (x, centerline_y),
                (min(x + dash_length, self.screen_width), centerline_y),
                width=2
            )
            x += dash_length + gap_length

    def draw_lidar_sensor(self):
        """Draw clean, non-obstructing circular LiDAR perception zone."""
        v = self.env.vehicle
        center_x = self.to_screen_x(v.center[0])
        center_y = self.to_screen_y(v.center[1])
        lidar_px_radius = int(self.env.lidar.radius * self.scale)

        # Ultra-subtle sheer aura (does not block or hide anything underneath)
        pygame.draw.circle(
            self.alpha_surface,
            self.COLOR_LIDAR_AURA,
            (center_x, center_y),
            lidar_px_radius
        )

        # Crisp electric cyan boundary ring
        pygame.draw.circle(
            self.screen,
            self.COLOR_LIDAR_RING,
            (center_x, center_y),
            lidar_px_radius,
            width=2
        )

        # Dynamic LiDAR distance measurement lines to all detected obstacles
        detections = self.env.detections
        if detections is not None and len(detections["indices"]) > 0:
            peds = self.env.pedestrians
            for idx in detections["indices"]:
                ped_px = self.to_screen_x(peds.x[idx])
                ped_py = self.to_screen_y(peds.y[idx])
                # Semi-transparent cyan laser measurement line
                pygame.draw.line(self.alpha_surface, (0, 229, 255, 120), (center_x, center_y), (ped_px, ped_py), 2)

    def draw_scenarios(self, scenarios, detected_indices, ped_positions, stride=1):
        """Draw future trajectory scenario candidate fans for detected pedestrians."""
        if scenarios is None or len(scenarios) == 0:
            return 0

        D, S, horizon, _ = scenarios.shape
        drawn_count = 0

        for d in range(D):
            ped_idx = detected_indices[d]
            p0 = ped_positions[ped_idx]

            for s in range(0, S, stride):
                traj = scenarios[d, s]
                full_path = np.vstack([p0[None, :], traj])

                pixel_pts = self.to_screen_coords(full_path)
                points_list = [tuple(pt) for pt in pixel_pts]

                # Draw smooth glowing line strip onto transparent alpha surface
                pygame.draw.lines(
                    self.alpha_surface,
                    self.COLOR_TRAJ_LINE,
                    False,
                    points_list,
                    width=2
                )
                drawn_count += 1

        return drawn_count

    def draw_vehicle(self):
        """Draw ego vehicle body with roof detail and front directional indicator."""
        v = self.env.vehicle

        car_left = self.to_screen_x(v.x)
        car_top = self.to_screen_y(v.y + v.width / 2.0)
        car_w = int(v.length * self.scale)
        car_h = int(v.width * self.scale)

        # Main chassis
        pygame.draw.rect(
            self.screen,
            self.COLOR_CAR,
            (car_left, car_top, car_w, car_h),
            border_radius=2 if car_w < 10 else 4
        )
        pygame.draw.rect(
            self.screen,
            self.COLOR_CAR_OUTLINE,
            (car_left, car_top, car_w, car_h),
            width=1 if car_w < 10 else 2,
            border_radius=2 if car_w < 10 else 4
        )

        if car_w >= 14:
            # Full roof glass highlight for larger vehicles
            roof_inset_x = int(car_w * 0.22)
            roof_inset_y = int(car_h * 0.18)
            pygame.draw.rect(
                self.screen,
                self.COLOR_CAR_ROOF,
                (car_left + roof_inset_x, car_top + roof_inset_y, car_w - 2 * roof_inset_x, car_h - 2 * roof_inset_y),
                border_radius=2
            )

            # Front headlights / direction indicator
            front_x = car_left + car_w - 2
            pygame.draw.rect(
                self.screen,
                self.COLOR_HEADLIGHT,
                (front_x, car_top + 2, 3, 4),
                border_radius=1
            )
            pygame.draw.rect(
                self.screen,
                self.COLOR_HEADLIGHT,
                (front_x, car_top + car_h - 6, 3, 4),
                border_radius=1
            )
        else:
            # Compact pod styling: glowing front edge indicator
            front_x = car_left + car_w - 1
            pygame.draw.line(
                self.screen,
                self.COLOR_HEADLIGHT,
                (front_x, car_top + 2),
                (front_x, car_top + car_h - 2),
                width=2
            )

        # Speed badge above vehicle
        car_text = f"v={v.speed_real:.1f} m/s | a={v.last_accel_real:+.1f} m/s²"
        txt_surf = self.font_labels.render(car_text, True, (15, 23, 42))
        badge_rect = pygame.Rect(car_left, max(4, car_top - 18), txt_surf.get_width() + 6, 15)
        pygame.draw.rect(self.screen, (255, 255, 255, 220), badge_rect, border_radius=3)
        pygame.draw.rect(self.screen, (150, 160, 175), badge_rect, width=1, border_radius=3)
        self.screen.blit(txt_surf, (car_left + 3, max(4, car_top - 18) + 1))

    def draw_pedestrians(self):
        """Draw all pedestrians with high-contrast state styling, LiDAR distance badges, and clear badges."""
        peds = self.env.pedestrians
        detections = self.env.detections
        detected_mask = detections["mask"] if detections is not None else [False] * peds.n
        distances = detections["distances"] if (detections is not None and "distances" in detections) else None

        ped_speeds = peds.speed_real

        for i in range(peds.n):
            px = self.to_screen_x(peds.x[i])
            py = self.to_screen_y(peds.y[i])
            is_detected = detected_mask[i]

            if is_detected:
                dist_m = float(distances[i]) if distances is not None else 0.0
                # Detected Pedestrian: Larger (6.5px), glowing neon emerald with double ring
                ped_radius = 6
                # Outer dark ring
                pygame.draw.circle(self.screen, self.COLOR_PED_OUTLINE, (px, py), ped_radius + 2)
                # Outer highlight
                pygame.draw.circle(self.screen, self.COLOR_PED_DETECTED_RING, (px, py), ped_radius + 1)
                # Neon core
                pygame.draw.circle(self.screen, self.COLOR_PED_DETECTED, (px, py), ped_radius)

                # High-contrast distance + speed badge: e.g. "d: 7.4m | 1.2 m/s"
                label_text = f"d:{dist_m:.1f}m | {ped_speeds[i]:.1f}m/s"
                txt_surf = self.font_labels.render(label_text, True, (6, 78, 59))
                pill_w = txt_surf.get_width() + 8
                pill_h = 14
                pill_x = px - pill_w // 2
                pill_y = py - ped_radius - 17

                # Glowing mint badge for detected threats
                pygame.draw.rect(self.screen, (236, 253, 245), (pill_x, pill_y, pill_w, pill_h), border_radius=4)
                pygame.draw.rect(self.screen, (16, 185, 129), (pill_x, pill_y, pill_w, pill_h), width=1, border_radius=4)
                self.screen.blit(txt_surf, (pill_x + 4, pill_y + 1))
            else:
                # Undetected Pedestrian: Coral red with dark outline
                ped_radius = 5
                pygame.draw.circle(self.screen, self.COLOR_PED_OUTLINE, (px, py), ped_radius + 1)
                pygame.draw.circle(self.screen, self.COLOR_PED_NORMAL, (px, py), ped_radius)

                label_text = f"{ped_speeds[i]:.1f} m/s"
                txt_surf = self.font_labels.render(label_text, True, (15, 23, 42))
                pill_w = txt_surf.get_width() + 6
                pill_h = 13
                pill_x = px - pill_w // 2
                pill_y = py - ped_radius - 15

                pygame.draw.rect(self.screen, (255, 255, 255, 230), (pill_x, pill_y, pill_w, pill_h), border_radius=3)
                pygame.draw.rect(self.screen, (160, 170, 180), (pill_x, pill_y, pill_w, pill_h), width=1, border_radius=3)
                self.screen.blit(txt_surf, (pill_x + 3, pill_y + 1))

    def draw_hud(self, current_cycle=1, total_cycles=1, detected_count=0, num_scenarios_drawn=0,
                 current_accel=0.0, solve_time_ms=0.0, paused=False):
        """Draw top telemetry bar with speed, acceleration, solve latency, and status."""
        v = self.env.vehicle

        status_text = "PAUSED [Press Space]" if paused else "LIVE [Press Space]"
        status_color = (220, 38, 38) if paused else (22, 163, 74)

        accel_str = f"Accel: {current_accel:+.2f} m/s²"
        solve_str = f"WDRO: {solve_time_ms:.0f}ms" if detected_count > 0 else "WDRO: Cruise"

        detections = self.env.detections
        if detected_count > 0 and detections is not None and len(detections["indices"]) > 0:
            min_dist = float(np.min(detections["distances"][detections["indices"]]))
            threat_str = f"LiDAR Threats: {detected_count} (min {min_dist:.1f}m)"
        else:
            threat_str = f"LiDAR Threats: {detected_count}"

        hud_line = (
            f"Cycle: {current_cycle}/{total_cycles} | "
            f"Speed: {v.speed_real:.1f} m/s ({v.speed_real * 3.6:.1f} km/h) | "
            f"{accel_str} | {solve_str} | "
            f"{threat_str} | "
        )

        hud_surf = self.font_hud.render(hud_line, True, self.COLOR_TEXT)
        status_surf = self.font_hud.render(status_text, True, status_color)

        banner_rect = pygame.Rect(10, 8, hud_surf.get_width() + status_surf.get_width() + 24, 26)
        pygame.draw.rect(self.screen, (255, 255, 255), banner_rect, border_radius=5)
        pygame.draw.rect(self.screen, (203, 213, 225), banner_rect, width=1, border_radius=5)

        self.screen.blit(hud_surf, (16, 13))
        self.screen.blit(status_surf, (16 + hud_surf.get_width(), 13))

    def draw_planned_acceleration_profile(self, planned_accels=None):
        """Draw interactive HUD widget showing the planned acceleration sequence across N steps."""
        panel_w = 380
        panel_h = 102
        panel_x = self.screen_width - panel_w - 12
        panel_y = 8

        # Panel background and clean rounded border
        card_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=6)
        pygame.draw.rect(self.screen, (203, 213, 225), card_rect, width=1, border_radius=6)

        # Title
        title_surf = self.font_plan.render("MPC Planned Accel Sequence (N=20 steps, 4.0s)", True, (15, 23, 42))
        self.screen.blit(title_surf, (panel_x + 12, panel_y + 8))

        if planned_accels is None:
            # Cruising status
            status_text = "Status: Cruising (Tracking target v_des = 8.0 m/s)"
            status_surf = self.font_labels.render(status_text, True, (100, 116, 139))
            self.screen.blit(status_surf, (panel_x + 12, panel_y + 36))

            # Flat zero line
            line_y = panel_y + 68
            pygame.draw.line(self.screen, (203, 213, 225), (panel_x + 12, line_y), (panel_x + panel_w - 12, line_y), 1)
            lbl = self.font_labels.render("a = 0.0 m/s²", True, (148, 163, 184))
            self.screen.blit(lbl, (panel_x + panel_w // 2 - 28, line_y + 4))
            return

        accels_arr = np.asarray(planned_accels, dtype=float)
        N = len(accels_arr)
        a0 = accels_arr[0]
        min_a = np.min(accels_arr)
        max_a = np.max(accels_arr)

        # Subtitle metrics
        metric_str = f"Active a0*: {a0:+.2f} m/s² | Min: {min_a:+.2f} | Max: {max_a:+.2f}"
        metric_color = (220, 38, 38) if a0 < -0.3 else ((22, 163, 74) if a0 > 0.3 else (71, 85, 105))
        metric_surf = self.font_labels.render(metric_str, True, metric_color)
        self.screen.blit(metric_surf, (panel_x + 12, panel_y + 26))

        # Chart area setup
        chart_left = panel_x + 16
        chart_right = panel_x + panel_w - 16
        chart_w = chart_right - chart_left
        zero_y = panel_y + 66  # baseline for a = 0

        # Draw zero line
        pygame.draw.line(self.screen, (203, 213, 225), (chart_left - 4, zero_y), (chart_right + 4, zero_y), 1)

        # Scale: max magnitude 8.0 m/s^2 maps to 24 pixels
        px_per_accel = 24.0 / 8.0

        bar_slot_w = chart_w / N
        bar_w = max(4, int(bar_slot_w - 3))

        for k in range(N):
            val = accels_arr[k]
            bx = int(chart_left + k * bar_slot_w + (bar_slot_w - bar_w) / 2)
            bar_len = int(abs(val) * px_per_accel)
            bar_len = max(2, min(bar_len, 26))

            if val < -0.05:
                # Braking (red bar going down)
                by = zero_y
                color = (239, 68, 68) if k > 0 else (220, 38, 38)
            elif val > 0.05:
                # Accelerating (green bar going up)
                by = zero_y - bar_len
                color = (16, 185, 129) if k > 0 else (22, 163, 74)
            else:
                by = zero_y - 1
                bar_len = 2
                color = (148, 163, 184)

            pygame.draw.rect(self.screen, color, (bx, by, bar_w, bar_len), border_radius=1)

            # Step 0 (active command) indicator outline
            if k == 0:
                highlight_rect = pygame.Rect(bx - 1, min(by, zero_y) - 2, bar_w + 2, bar_len + 4)
                pygame.draw.rect(self.screen, (15, 23, 42), highlight_rect, width=1, border_radius=2)

        # Horizon labels: k=0 (Now), k=10 (2s), k=19 (4s)
        lbl0 = self.font_labels.render("k=0 (Now)", True, (15, 23, 42))
        lbl_mid = self.font_labels.render("k=10 (2s)", True, (100, 116, 139))
        lbl_end = self.font_labels.render("k=19 (4s)", True, (100, 116, 139))
        self.screen.blit(lbl0, (chart_left, panel_y + panel_h - 16))
        self.screen.blit(lbl_mid, (chart_left + chart_w // 2 - 20, panel_y + panel_h - 16))
        self.screen.blit(lbl_end, (chart_right - lbl_end.get_width(), panel_y + panel_h - 16))

    def draw(self, scenarios=None, detected_indices=None, ped_positions=None,
             current_cycle=1, total_cycles=1, detected_count=0,
             current_accel=0.0, solve_time_ms=0.0, planned_accels=None, paused=False):
        """Render complete frame with strict layer ordering for maximum clarity."""
        # 1. Reset alpha surface
        self.alpha_surface.fill((0, 0, 0, 0))

        # 2. Layer 1: Road & Sidewalks
        self.draw_static_world()

        # 3. Layer 2: LiDAR sensor ring
        self.draw_lidar_sensor()

        # 4. Layer 3: Future Trajectory Scenario Fans
        num_scenarios_drawn = 0
        if scenarios is not None and detected_indices is not None and len(detected_indices) > 0:
            num_scenarios_drawn = self.draw_scenarios(scenarios, detected_indices, ped_positions)

        # 5. Blit alpha surface (aura + trajectory lines)
        self.screen.blit(self.alpha_surface, (0, 0))

        # 6. Layer 4: Ego Vehicle
        self.draw_vehicle()

        # 7. Layer 5: Pedestrians (drawn on TOP of trajectories and LiDAR)
        self.draw_pedestrians()

        # 8. Layer 6: HUD Banner
        self.draw_hud(
            current_cycle=current_cycle,
            total_cycles=total_cycles,
            detected_count=detected_count,
            num_scenarios_drawn=num_scenarios_drawn,
            current_accel=current_accel,
            solve_time_ms=solve_time_ms,
            paused=paused
        )

        # 9. Layer 7: Planned Acceleration Sequence HUD Card
        self.draw_planned_acceleration_profile(planned_accels=planned_accels)

        # Double-buffer flip
        pygame.display.flip()

    def close(self):
        """Clean up Pygame window."""
        pygame.quit()

