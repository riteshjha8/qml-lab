
from pathlib import Path
from PIL import Image, ImageOps
import shutil
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "datasetofdigit"
OUTPUT = ROOT / "filtered_dataset"
REPORT = ROOT / "filter_report.csv"

VALID_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

# Tune only after inspecting the actual images.
INK_THRESHOLD = 220
MIN_INK_PIXELS = 5

if not SOURCE.is_dir():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

OUTPUT.mkdir(parents=True, exist_ok=True)

records = []
accepted = 0
rejected = 0

for digit in range(10):
    source_folder = SOURCE / str(digit)
    output_folder = OUTPUT / str(digit)

    if not source_folder.is_dir():
        records.append({
            "file": str(source_folder),
            "label": digit,
            "status": "rejected",
            "reason": "Missing class folder"
        })
        rejected += 1
        continue

    for path in source_folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            records.append({
                "file": str(path),
                "label": digit,
                "status": "rejected",
                "reason": "Unsupported file format"
            })
            rejected += 1
            continue

        try:
            with Image.open(path) as original:
                original.verify()

            with Image.open(path) as original:
                img = ImageOps.exif_transpose(original).convert("L")
                pixels = np.asarray(img)

            ink_pixels = int(np.sum(pixels < INK_THRESHOLD))

            if ink_pixels < MIN_INK_PIXELS:
                raise ValueError("Blank or nearly blank image")

            # Preserve the original image file and format.
            relative_path = path.relative_to(source_folder)
            destination = output_folder / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

            records.append({
                "file": str(path),
                "label": digit,
                "status": "accepted",
                "reason": ""
            })
            accepted += 1

        except Exception as exc:
            records.append({
                "file": str(path),
                "label": digit,
                "status": "rejected",
                "reason": str(exc)
            })
            rejected += 1

with REPORT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["file", "label", "status", "reason"]
    )
    writer.writeheader()
    writer.writerows(records)

print("Dataset filtering completed.")
print("Accepted files:", accepted)
print("Rejected files:", rejected)
print("Filtered dataset:", OUTPUT)
print("Quality report:", REPORT)
