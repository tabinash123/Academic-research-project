import os
import sys
from pathlib import Path
import numpy as np

# Ensure local directory and environment_setup are importable
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir / "environment_setup"))

from training_pedestrian import PedestrianGroup
from units import DT


def collect_dataset(
    num_samples=100,
    num_pedestrians=10,
    horizon=20,
    road_y=20,
    road_width=12,
    world_width=120,
    out_file="pedestrian_dataset.npy",
    seed=42,
):
    """Collects a dataset of pedestrian trajectories for DR-MPC residual fitting.

    Simulates active pedestrian crowds across batches to collect independent
    trajectory samples. Each sample spans (horizon + 1) states over (horizon * DT) seconds.

    Parameters
    ----------
    num_samples : int
        Total number of trajectory samples to collect (default: 100).
    num_pedestrians : int
        Number of pedestrians active in the simulation crowd (default: 10).
    horizon : int
        Number of future prediction steps (default: 20 steps = 4.0s at DT=0.2s).
    road_y : float
        Y-center of the road.
    road_width : float
        Width of the road.
    world_width : float
        Width of the simulation world.
    out_file : str
        Filename / path to save the .npy dataset.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dataset : ndarray, shape (num_samples, horizon + 1, 4)
        Collected trajectories containing [x, y, vx, vy].
    """
    total_timesteps = horizon + 1  # t=0 (initial state) + horizon future steps
    duration_s = horizon * DT
    print(f"==================================================")
    print(f"Collecting Historical Pedestrian Dataset")
    print(f"Total Samples (S):     {num_samples}")
    print(f"Active Crowd (N_ped):  {num_pedestrians}")
    print(f"Horizon Steps (N):     {horizon} steps ({duration_s:.1f} s at DT={DT}s)")
    print(f"Timesteps per Sample:  {total_timesteps}")
    print(f"==================================================")

    dataset = np.zeros((num_samples, total_timesteps, 4))
    samples_collected = 0
    batch_idx = 0
    rng = np.random.default_rng(seed)

    while samples_collected < num_samples:
        # Determine how many pedestrians to take from this batch
        count = min(num_pedestrians, num_samples - samples_collected)
        batch_seed = int(rng.integers(0, 1_000_000))

        pedestrians = PedestrianGroup(
            num_pedestrians=num_pedestrians,
            road_y=road_y,
            road_width=road_width,
            world_width=world_width,
            seed=batch_seed,
        )

        batch_data = np.zeros((count, total_timesteps, 4))

        for t in range(total_timesteps):
            # Extract states for the first `count` pedestrians
            batch_data[:, t, 0] = pedestrians.x[:count]
            batch_data[:, t, 1] = pedestrians.y[:count]

            speed_real = pedestrians.speed_real[:count]
            heading = pedestrians.heading[:count]
            batch_data[:, t, 2] = speed_real * np.cos(heading)  # vx
            batch_data[:, t, 3] = speed_real * np.sin(heading)  # vy

            if t < horizon:
                pedestrians.step()

        dataset[samples_collected : samples_collected + count] = batch_data
        samples_collected += count
        batch_idx += 1
        print(f"Batch {batch_idx}: Collected {samples_collected}/{num_samples} samples...")

    # Determine save path (save in script's directory if relative path)
    if not os.path.isabs(out_file):
        save_path = current_dir / out_file
    else:
        save_path = Path(out_file)

    print(f"Saving dataset of shape {dataset.shape} to '{save_path}'...")
    np.save(save_path, dataset)
    print("Done! Dataset is ready.")
    return dataset


if __name__ == "__main__":
    collect_dataset()
