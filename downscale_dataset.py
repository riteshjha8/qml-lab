
from pathlib import Path
from collections import Counter
import csv

import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "filtered_dataset"
OUTPUT = ROOT / "downscaled_dataset"
SIZE = 16

VALID_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

if not SOURCE.is_dir():
    raise FileNotFoundError(
        f"Filtered dataset not found: {SOURCE.resolve()}"
    )

OUTPUT.mkdir(parents=True, exist_ok=True)

records = []
class_counts = Counter()
all_images = []
all_labels = []
failed = 0

# --------------------------------------------------
# Step 3: Downscale images class by class
# --------------------------------------------------
for digit in range(10):
    source_folder = SOURCE / str(digit)
    output_folder = OUTPUT / str(digit)

    if not source_folder.is_dir():
        print(f"WARNING: Missing class folder {digit}")
        continue

    for path in sorted(source_folder.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            # Read and standardize image orientation.
            with Image.open(path) as original:
                img = ImageOps.exif_transpose(original).convert("L")

            # Identify the handwritten ink.
            gray = np.asarray(img, dtype=np.uint8)
            ink_mask = gray < 220

            if not ink_mask.any():
                raise ValueError("No dark pixels found")

            # Crop around the digit.
            ys, xs = np.where(ink_mask)
            crop = img.crop((
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1
            ))

            # Fit inside a 14x14 area without distortion.
            # Leave a 1-pixel margin in the 16x16 canvas.
            crop.thumbnail(
                (SIZE - 2, SIZE - 2),
                Image.Resampling.LANCZOS
            )

            # Center on a white square canvas.
            canvas = Image.new("L", (SIZE, SIZE), 255)
            x = (SIZE - crop.width) // 2
            y = (SIZE - crop.height) // 2
            canvas.paste(crop, (x, y))

            # Save the downscaled grayscale image.
            relative_path = path.relative_to(source_folder)
            destination = output_folder / relative_path
            destination = destination.with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(destination)

            # Normalize to [0, 1], with ink near 0.
            normalized = (
                255.0 - np.asarray(canvas, dtype=np.float32)
            ) / 255.0

            all_images.append(normalized)
            all_labels.append(digit)
            class_counts[digit] += 1

            records.append({
                "source": str(path),
                "output": str(destination),
                "label": digit,
                "width": canvas.width,
                "height": canvas.height,
                "status": "success"
            })

        except Exception as exc:
            failed += 1
            records.append({
                "source": str(path),
                "output": "",
                "label": digit,
                "width": "",
                "height": "",
                "status": f"failed: {exc}"
            })

# --------------------------------------------------
# Save labels, normalized arrays, and processing report
# --------------------------------------------------
if not all_images:
    raise RuntimeError("No images were successfully processed.")

X = np.stack(all_images).astype(np.float32)
y = np.asarray(all_labels, dtype=np.int64)

np.savez_compressed(
    OUTPUT / "downscaled_data.npz",
    X=X,
    y=y,
    image_size=SIZE
)

with open(
    OUTPUT / "downscaling_report.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "source", "output", "label",
            "width", "height", "status"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

# --------------------------------------------------
# Report
# --------------------------------------------------
print("\nSTEP 3: DOWNSCALING RESULTS")
print("-" * 40)
print("Successfully processed:", len(X))
print("Failed during processing:", failed)
print("Output shape:", X.shape)
print("Label shape:", y.shape)
print("Pixel range:", float(X.min()), "to", float(X.max()))
print("Output directory:", OUTPUT.resolve())

print("\nImages per digit:")
for digit in range(10):
    print(f"Digit {digit}: {class_counts[digit]}")

print("\nSaved:")
print(" - downscaled_data.npz")
print(" - downscaling_report.csv")
print(" - Individual 16x16 PNG images")
