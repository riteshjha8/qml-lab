
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "downscaled_dataset_28" / "downscaled_data_28.npz"
OUTPUT_DIR = ROOT / "inspection_results_28"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]

print("28x28 DATASET QUALITY CHECK")
print("---------------------------")
print("Image shape:", X.shape)
print("Labels shape:", y.shape)
print("Pixel range:", float(X.min()), "to", float(X.max()))
print("Non-finite values:", int((~np.isfinite(X)).sum()))
print("Completely blank images:", int(np.sum(X.sum(axis=(1, 2)) == 0)))
print("Images with fewer than 5 nonzero pixels:",
      int(np.sum(np.count_nonzero(X > 0, axis=(1, 2)) < 5)))

print("\nImages per digit:")
for digit in range(10):
    print(digit, ":", int(np.sum(y == digit)))

# Display a sample from each digit.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for digit, ax in enumerate(axes.flat):
    indices = np.flatnonzero(y == digit)
    idx = indices[len(indices) // 2]

    ax.imshow(X[idx], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Digit {digit}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "digit_samples_28.png", dpi=150)
plt.show()

print("\nSaved sample image:",
      OUTPUT_DIR / "digit_samples_28.png")
