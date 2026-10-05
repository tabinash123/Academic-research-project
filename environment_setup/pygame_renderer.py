import pygame
import numpy as np


class PygameRenderer:
    """Hardware-accelerated real-time visualizer for 2D autonomous driving simulation.

    Reads Environment state and draws it directly to a native window.
    Contains no physics of its own.
    """

    def __init__(self, env, scale=10, title="Real-Time Autonomous Driving Simulation"):
        self.env = env
        self.scale = scale

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

        # UI & Telemetry
        self.COLOR_TEXT = (15, 23, 42)
        self.font_hud = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_labels = pygame.font.SysFont("Arial", 10, bold=True)

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

    def draw_static_world(self):
        """Draw grass sidewalks, asphalt road surface, curbs, and dashed centerline."""
        self.screen.fill(self.COLOR_GRASS)

        road_top_real = self.env.road_y + self.env.road_width / 2.0
        road_screen_y = self.to_screen_y(road_top_real)
        road_screen_height = int(self.env.road_width * self.scale)

        pygame.draw.rect(
            self.screen,
            self.COLOR_ROAD,
            (0, road_screen_y, self.screen_width, road_screen_height)
        )

        # Curbs
        pygame.draw.line(self.screen, self.COLOR_CURB, (0, road_screen_y), (self.screen_width, road_screen_y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURB, (0, road_screen_y + road_screen_height), (self.screen_width, road_screen_y + road_screen_height), 2)

        # Dashed centerline
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
        """Draw clean circular LiDAR perception zone."""
        v = self.env.vehicle
        center_x = self.to_screen_x(v.center[0])
        center_y = self.to_screen_y(v.center[1])
        lidar_px_radius = int(self.env.lidar.radius * self.scale)

        pygame.draw.circle(
            self.alpha_surface,
            self.COLOR_LIDAR_AURA,
            (center_x, center_y),
            lidar_px_radius
        )
        pygame.draw.circle(
            self.screen,
            self.COLOR_LIDAR_RING,
            (center_x, center_y),
            lidar_px_radius,
            width=2
        )

    def draw_vehicle(self):
        """Draw ego vehicle chassis, roof, headlights, and speed label."""
        v = self.env.vehicle

        car_left = self.to_screen_x(v.x)
        car_top = self.to_screen_y(v.y + v.width / 2.0)
        car_w = int(v.length * self.scale)
        car_h = int(v.width * self.scale)

        pygame.draw.rect(
            self.screen,
            self.COLOR_CAR,
            (car_left, car_top, car_w, car_h),
            border_radius=4
        )
        pygame.draw.rect(
            self.screen,
            self.COLOR_CAR_OUTLINE,
            (car_left, car_top, car_w, car_h),
            width=2,
            border_radius=4
        )

        # Roof
        roof_inset_x = int(car_w * 0.22)
        roof_inset_y = int(car_h * 0.18)
        pygame.draw.rect(
            self.screen,
            self.COLOR_CAR_ROOF,
            (car_left + roof_inset_x, car_top + roof_inset_y, car_w - 2 * roof_inset_x, car_h - 2 * roof_inset_y),
            border_radius=2
        )

        # Headlights
        front_x = car_left + car_w - 2
        pygame.draw.rect(self.screen, self.COLOR_HEADLIGHT, (front_x, car_top + 2, 3, 4), border_radius=1)
        pygame.draw.rect(self.screen, self.COLOR_HEADLIGHT, (front_x, car_top + car_h - 6, 3, 4), border_radius=1)

        car_text = f"v={v.speed_real:.1f} m/s | a={v.last_accel_real:+.1f} m/s²"
        txt_surf = self.font_labels.render(car_text, True, (15, 23, 42))
        badge_rect = pygame.Rect(car_left, max(4, car_top - 18), txt_surf.get_width() + 6, 15)
        pygame.draw.rect(self.screen, (255, 255, 255, 220), badge_rect, border_radius=3)
        pygame.draw.rect(self.screen, (150, 160, 175), badge_rect, width=1, border_radius=3)
        self.screen.blit(txt_surf, (car_left + 3, max(4, car_top - 18) + 1))

    def draw_pedestrians(self):
        """Draw pedestrians with clear contrast badges."""
        peds = self.env.pedestrians
        detections = self.env.detections
        detected_mask = detections["mask"] if detections is not None else [False] * peds.n
        ped_speeds = peds.speed_real

        for i in range(peds.n):
            px = self.to_screen_x(peds.x[i])
            py = self.to_screen_y(peds.y[i])
            is_detected = detected_mask[i]

            if is_detected:
                ped_radius = 6
                pygame.draw.circle(self.screen, self.COLOR_PED_OUTLINE, (px, py), ped_radius + 2)
                pygame.draw.circle(self.screen, self.COLOR_PED_DETECTED_RING, (px, py), ped_radius + 1)
                pygame.draw.circle(self.screen, self.COLOR_PED_DETECTED, (px, py), ped_radius)
            else:
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

    def draw_hud(self, current_cycle=1, total_cycles=1, paused=False):
        """Draw top telemetry bar and controls status."""
        v = self.env.vehicle
        detected_count = len(self.env.detections["indices"]) if self.env.detections is not None else 0

        status_text = "PAUSED [Press Space to Resume]" if paused else "LIVE [Press Space to Pause]"
        status_color = (220, 38, 38) if paused else (22, 163, 74)

        hud_line = (
            f"Cycle: {current_cycle}/{total_cycles} | "
            f"Vehicle: {v.speed_real:.1f} m/s ({v.speed_real * 3.6:.1f} km/h) | "
            f"Accel: {v.last_accel_real:+.1f} m/s² | "
            f"LiDAR Detections: {detected_count} | "
        )

        hud_surf = self.font_hud.render(hud_line, True, self.COLOR_TEXT)
        status_surf = self.font_hud.render(status_text, True, status_color)

        banner_rect = pygame.Rect(10, 8, hud_surf.get_width() + status_surf.get_width() + 24, 26)
        pygame.draw.rect(self.screen, (255, 255, 255), banner_rect, border_radius=5)
        pygame.draw.rect(self.screen, (203, 213, 225), banner_rect, width=1, border_radius=5)

        self.screen.blit(hud_surf, (16, 13))
        self.screen.blit(status_surf, (16 + hud_surf.get_width(), 13))

    def draw(self, current_cycle=1, total_cycles=1, paused=False):
        """Render complete simulation frame."""
        self.alpha_surface.fill((0, 0, 0, 0))

        self.draw_static_world()
        self.draw_lidar_sensor()
        self.screen.blit(self.alpha_surface, (0, 0))

        self.draw_vehicle()
        self.draw_pedestrians()
        self.draw_hud(current_cycle, total_cycles, paused)

        pygame.display.flip()

    def close(self):
        """Clean up Pygame window."""
        pygame.quit()
