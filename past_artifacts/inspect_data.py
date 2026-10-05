import os
from pathlib import Path
import numpy as np

current_dir = Path(__file__).resolve().parent
dataset_path = current_dir / "pedestrian_dataset.npy"

if not dataset_path.exists():
    print(f"File not found: {dataset_path}")
    print("Please run collect_data.py first!")
    exit(1)

# Load binary numpy file
dataset = np.load(dataset_path)
S, T, features = dataset.shape
DT = 0.2

print("========== DATASET INFO ==========")
print(f"Shape: {dataset.shape}")
print(f"Dimensions: (Samples={S}, Timesteps={T}, Features={features})")
print(f"Time: Duration={(T-1)*DT:.1f}s (DT={DT}s)")
print("Features are: [x, y, vx, vy]\n")

print("========== SAMPLE 0 (First 5 Timesteps) ==========")
sample_0_early = dataset[0, :5, :]

for t, state in enumerate(sample_0_early):
    x, y, vx, vy = state
    print(f"t={t:02d} ({t*DT:.1f}s): x={x:6.2f}, y={y:6.2f}, vx={vx:5.2f}, vy={vy:5.2f}")

print("\n(Dataset inspection completed successfully.)")
