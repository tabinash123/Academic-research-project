import numpy as np


class Lidar:
    """Simple circular-radius pedestrian detector.

    Pure geometry -- returns detection data (indices, positions,
    distances) that a controller could consume directly. Rendering
    (coloring detected pedestrians) is the Renderer's job, not this
    class's.
    """

    def __init__(self, radius):
        self.radius = radius

    def detect(self, vehicle_center, pedestrian_positions):
        vx, vy = vehicle_center

        distances = np.sqrt(
            (pedestrian_positions[:, 0] - vx) ** 2 +
            (pedestrian_positions[:, 1] - vy) ** 2
        )

        detected_mask = distances <= self.radius
        detected_indices = np.where(detected_mask)[0]
        detected_positions = pedestrian_positions[detected_mask]

        return {
            "mask": detected_mask,
            "indices": detected_indices,
            "positions": detected_positions,
            "distances": distances,
        }
