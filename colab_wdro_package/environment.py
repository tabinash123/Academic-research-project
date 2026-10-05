from vehicle import Vehicle
from pedestrian import PedestrianGroup
from lidar import Lidar


class Environment:
    """Owns all simulation state and physics.

    No matplotlib import, no rendering. This is the piece a future
    controller calls into (env.vehicle.move(accel_real=...) is invoked
    externally by whatever is running the control loop) -- Environment
    itself stays controller-agnostic.
    """

    def __init__(self,
                 world_width=120, world_height=40,
                 road_y=20, road_width=12,
                 num_pedestrians=5,
                 car_x=5, car_speed_real=8.0, car_length=0.5, car_width=2,
                 lidar_radius=12):

        self.world_width = world_width
        self.world_height = world_height
        self.road_y = road_y
        self.road_width = road_width

        self.vehicle = Vehicle(
            x=car_x, y=road_y, speed_real=car_speed_real,
            length=car_length, width=car_width,
            world_width=world_width,
            road_y=road_y, road_width=road_width
        )

        self.pedestrians = PedestrianGroup(
            num_pedestrians=num_pedestrians,
            road_y=road_y, road_width=road_width,
            world_width=world_width
        )

        self.lidar = Lidar(radius=lidar_radius)

        self.detections = None

    def step(self, accel_real=0.0):
        """Advance the whole simulation by one timestep.

        accel_real: acceleration to apply to the vehicle this step
        (defaults to 0 -- constant speed). A controller running outside
        Environment computes this and passes it in.
        """

        self.vehicle.move(accel_real=accel_real)
        self.pedestrians.step()

        self.detections = self.lidar.detect(
            self.vehicle.center,
            self.pedestrians.positions
        )

        return self.detections
