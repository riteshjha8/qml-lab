
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------
# STEP 4: Inspect duplicate and potentially ambiguous
# handwritten-digit images.
# Original dataset and Step 3 files remain unchanged.
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "downscaled_dataset" / "downscaled_data.npz"
OUTPUT_DIR = ROOT / "inspection_results"
SHEET_DIR = OUTPUT_DIR / "review_sheets"

IMAGE_SIZE = 16
INK_THRESHOLD = 0.15
SHEET_COLUMNS = 10
THUMBNAIL_SIZE = 80
LABEL_HEIGHT = 18
SHEET_ROWS = 10

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Could not find: {DATA_FILE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHEET_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_FILE)
X = data["X"].astype(np.float32)
y = data["y"].astype(int)

if X.ndim != 3 or X.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
    raise ValueError(f"Unexpected image array shape: {X.shape}")

if len(X) != len(y):
    raise ValueError("Number of images does not match number of labels.")

if not np.isfinite(X).all():
    raise ValueError("Dataset contains NaN or infinite pixel values.")

if X.min() < 0 or X.max() > 1:
    raise ValueError("Expected normalized pixel values between 0 and 1.")

# Save a unique identifier for every sample.
# Exact pixel duplicates with different labels are suspicious.
hash_to_indices = defaultdict(list)

for i, image in enumerate(X):
    packed = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    signature = hashlib.sha256(packed.tobytes()).hexdigest()
    hash_to_indices[signature].append(i)

conflict_indices = set()
duplicate_rows = []

for signature, indices in hash_to_indices.items():
    labels = sorted(set(int(y[i]) for i in indices))

    if len(labels) > 1:
        conflict_indices.update(indices)

        for i in indices:
            duplicate_rows.append({
                "sample_index": i,
                "label": int(y[i]),
                "conflicting_labels": ",".join(map(str, labels)),
                "sha256": signature,
            })

duplicate_csv = OUTPUT_DIR / "conflicting_exact_duplicates.csv"

with duplicate_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_index",
            "label",
            "conflicting_labels",
            "sha256",
        ],
    )
    writer.writeheader()
    writer.writerows(duplicate_rows)

# Measure basic image properties. These are only screening
# signals; unusual values do not prove a label is wrong.
records = []

for i, (image, label) in enumerate(zip(X, y)):
    mask = image > INK_THRESHOLD
    ink_count = int(mask.sum())

    if ink_count:
        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        edge_touch = (
            x_min == 0 or x_max == IMAGE_SIZE - 1
            or y_min == 0 or y_max == IMAGE_SIZE - 1
        )

        center_x = float(xs.mean())
        center_y = float(ys.mean())
    else:
        width = height = 0
        edge_touch = False
        center_x = center_y = 0.0

    records.append({
        "sample_index": i,
        "label": int(label),
        "ink_pixels": ink_count,
        "bbox_width": width,
        "bbox_height": height,
        "touches_edge": edge_touch,
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "conflicting_exact_duplicate": i in conflict_indices,
    })

# Flag per-digit outliers using the interquartile range (IQR).
# Outliers are sent for human review, never deleted automatically.
review_indices = set(conflict_indices)
outlier_reasons = defaultdict(list)

for digit in range(10):
    digit_records = [r for r in records if r["label"] == digit]

    for metric in ("ink_pixels", "bbox_width", "bbox_height"):
        values = np.array([r[metric] for r in digit_records], dtype=float)

        if len(values) < 4:
            continue

        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        # Avoid treating a constant feature as an outlier detector.
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for record, value in zip(digit_records, values):
            if value < lower or value > upper:
                idx = record["sample_index"]
                review_indices.add(idx)
                outlier_reasons[idx].append(
                    f"{metric}_outlier"
                )

for record in records:
    idx = record["sample_index"]
    record["review_reasons"] = ";".join(
        sorted(set(outlier_reasons[idx]))
    )
    if idx in conflict_indices:
        record["review_reasons"] = (
            record["review_reasons"] + ";"
            if record["review_reasons"] else ""
        ) + "conflicting_exact_duplicate"

# Write a complete, auditable inspection report.
report_csv = OUTPUT_DIR / "image_inspection_report.csv"

with report_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

# Create contact sheets for the flagged samples.
# Each thumbnail is enlarged with nearest-neighbor interpolation
# so the 16x16 pixels remain visible.
flagged = sorted(review_indices)

for start in range(0, len(flagged), SHEET_COLUMNS * SHEET_ROWS):
    batch = flagged[start:start + SHEET_COLUMNS * SHEET_ROWS]
    sheet = Image.new(
        "RGB",
        (
            SHEET_COLUMNS * THUMBNAIL_SIZE,
            SHEET_ROWS * (THUMBNAIL_SIZE + LABEL_HEIGHT),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for position, idx in enumerate(batch):
        row = position // SHEET_COLUMNS
        col = position % SHEET_COLUMNS

        # X is inverted: ink is high, background is low.
        # Convert back to a conventional black-ink, white-background view.
        gray = np.clip(255 - X[idx] * 255, 0, 255).astype(np.uint8)
        thumbnail = Image.fromarray(gray, mode="L").resize(
            (THUMBNAIL_SIZE, THUMBNAIL_SIZE),
            Image.Resampling.NEAREST,
        ).convert("RGB")

        x = col * THUMBNAIL_SIZE
        y_pos = row * (THUMBNAIL_SIZE + LABEL_HEIGHT)

        sheet.paste(thumbnail, (x, y_pos))
        reason = ",".join(sorted(set(outlier_reasons[idx])))

        if idx in conflict_indices:
            reason = (reason + ",conflict").strip(",")

        label_text = f"#{idx} digit={y[idx]}"
        if reason:
            label_text += f" [{reason[:18]}]"

        draw.text(
            (x + 1, y_pos + THUMBNAIL_SIZE),
            label_text,
            fill="black",
        )

    sheet_number = start // (SHEET_COLUMNS * SHEET_ROWS) + 1
    sheet_path = SHEET_DIR / f"review_sheet_{sheet_number:03d}.png"
    sheet.save(sheet_path)

print("\nSTEP 4A: INSPECTION COMPLETED")
print("-" * 42)
print(f"Total images inspected: {len(X)}")
print(f"Exact-duplicate conflict samples: {len(conflict_indices)}")
print(f"Samples flagged for review: {len(review_indices)}")
print(f"Inspection report: {report_csv}")
print(f"Duplicate conflicts report: {duplicate_csv}")
print(f"Review sheets folder: {SHEET_DIR}")
print("\nNo images were deleted or relabeled.")
print("Review flags are candidates, not confirmed mistakes.")
