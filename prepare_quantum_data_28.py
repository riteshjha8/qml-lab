
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. Configuration and file paths
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_28_PATH = (
    ROOT / "downscaled_dataset_28" / "downscaled_data_28.npz"
)

DATA_16_PATH = (
    ROOT / "downscaled_dataset" / "downscaled_data.npz"
)

OLD_QUANTUM_PATH = (
    ROOT / "quantum_data" / "quantum_features.npz"
)

OUTPUT_DIR = ROOT / "quantum_data_28"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_COMPONENTS = 8
N_QUBITS = 8


# ============================================================
# 2. Load datasets
# ============================================================

print("Loading datasets...")

for path in [DATA_28_PATH, DATA_16_PATH, OLD_QUANTUM_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

data_28 = np.load(DATA_28_PATH)
data_16 = np.load(DATA_16_PATH)
old_quantum = np.load(OLD_QUANTUM_PATH)

X = data_28["X"].astype(np.float32)
y = data_28["y"].astype(np.int64)

old_y = data_16["y"].astype(np.int64)

print(f"28x28 images: {X.shape}")
print(f"Labels:       {y.shape}")


# ============================================================
# 3. Validate the 28x28 dataset
# ============================================================

if X.ndim != 3 or X.shape[1:] != (28, 28):
    raise ValueError(
        f"Expected image shape (N, 28, 28), got {X.shape}"
    )

if len(X) != len(y):
    raise ValueError("Image count and label count do not match.")

if not np.isfinite(X).all():
    raise ValueError("Images contain NaN or infinite values.")

if X.min() < 0.0 or X.max() > 1.0:
    raise ValueError(
        "Expected normalized image pixels between 0 and 1."
    )

# Image ordering and labels must remain consistent between datasets.
if not np.array_equal(y, old_y):
    raise ValueError(
        "Labels or image ordering differ between the 16x16 "
        "and 28x28 datasets. Check dataset ordering first."
    )

print("Dataset validation passed.")


# ============================================================
# 4. Reuse the original saved exclusions and split
# ============================================================

# IMPORTANT:
# We do NOT detect blank images again.
# We do NOT compare blank-image indices after resizing.
# We reuse the original saved indices for a consistent comparison.

excluded_indices = np.asarray(
    old_quantum["excluded_blank_indices"],
    dtype=np.int64,
)

original_train_indices = np.asarray(
    old_quantum["train_original_indices"],
    dtype=np.int64,
)

test_indices = np.asarray(
    old_quantum["test_original_indices"],
    dtype=np.int64,
)

all_indices = np.arange(len(y), dtype=np.int64)

# Check that saved indices are valid and contain no duplicates.
for name, indices in [
    ("excluded", excluded_indices),
    ("training", original_train_indices),
    ("test", test_indices),
]:
    if np.any(indices < 0) or np.any(indices >= len(y)):
        raise ValueError(f"{name} indices are out of range.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} indices contain duplicates.")

# Ensure no overlap between the saved partitions.
if np.intersect1d(
    original_train_indices, test_indices
).size > 0:
    raise ValueError("Original training and test sets overlap.")

if np.intersect1d(
    excluded_indices, original_train_indices
).size > 0:
    raise ValueError("Excluded indices overlap with training.")

if np.intersect1d(
    excluded_indices, test_indices
).size > 0:
    raise ValueError("Excluded indices overlap with testing.")

# Confirm the saved partitions cover all non-excluded samples.
valid_indices = np.setdiff1d(all_indices, excluded_indices)

saved_indices = np.sort(
    np.concatenate([original_train_indices, test_indices])
)

if not np.array_equal(saved_indices, valid_indices):
    raise ValueError(
        "Saved train/test indices do not cover exactly the "
        "non-excluded samples. Check the original split file."
    )

print(f"Original exclusions reused: {len(excluded_indices)}")
print(f"Original training samples:  {len(original_train_indices)}")
print(f"Test samples:               {len(test_indices)}")


# ============================================================
# 5. Create validation split
# ============================================================

# Validation is created from the original training partition only.
# The original test set remains untouched.

train_indices, val_indices = train_test_split(
    original_train_indices,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y[original_train_indices],
)

if np.intersect1d(train_indices, val_indices).size > 0:
    raise ValueError("Training and validation sets overlap.")

if np.intersect1d(train_indices, test_indices).size > 0:
    raise ValueError("Training and test sets overlap.")

if np.intersect1d(val_indices, test_indices).size > 0:
    raise ValueError("Validation and test sets overlap.")

print(f"Final training samples: {len(train_indices)}")
print(f"Validation samples:     {len(val_indices)}")
print(f"Final test samples:     {len(test_indices)}")


# ============================================================
# 6. Flatten 28x28 images
# ============================================================

X_flat = X.reshape(len(X), 28 * 28)

X_train = X_flat[train_indices]
X_val = X_flat[val_indices]
X_test = X_flat[test_indices]

y_train = y[train_indices]
y_val = y[val_indices]
y_test = y[test_indices]


# ============================================================
# 7. Fit PCA on training data only
# ============================================================

print("\nFitting PCA on training images...")

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE,
)

train_pca = pca.fit_transform(X_train)
val_pca = pca.transform(X_val)
test_pca = pca.transform(X_test)

explained_variance = float(
    np.sum(pca.explained_variance_ratio_)
)

print(
    f"PCA variance retained: {explained_variance:.4f} "
    f"({explained_variance * 100:.2f}%)"
)


# ============================================================
# 8. Convert PCA features to quantum rotation angles
# ============================================================

# Fit the scaler only on training features to prevent data leakage.

scaler = MinMaxScaler(
    feature_range=(0.0, np.pi) # type: ignore
)

train_angles = scaler.fit_transform(train_pca).astype(np.float32)
val_angles = scaler.transform(val_pca).astype(np.float32)
test_angles = scaler.transform(test_pca).astype(np.float32)


# ============================================================
# 9. Save quantum features
# ============================================================

output_npz = OUTPUT_DIR / "quantum_features_28.npz"

np.savez_compressed(
    output_npz,

    # Quantum input features
    train_angles=train_angles,
    val_angles=val_angles,
    test_angles=test_angles,

    # Labels
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,

    # Original sample indices
    train_original_indices=train_indices,
    val_original_indices=val_indices,
    test_original_indices=test_indices,

    # Original exclusion list; no new blank detection
    excluded_blank_indices=excluded_indices,

    # PCA information
    explained_variance_ratio=(
        pca.explained_variance_ratio_.astype(np.float32)
    ),
    explained_variance_total=np.float32(explained_variance),

    # Circuit configuration
    n_qubits=np.int64(N_QUBITS),
    n_components=np.int64(N_COMPONENTS),
    random_state=np.int64(RANDOM_STATE),
)

transformer_path = OUTPUT_DIR / "feature_transformers_28.joblib"

joblib.dump(
    {
        "pca": pca,
        "scaler": scaler,
    },
    transformer_path,
)


# ============================================================
# 10. Verify the saved files
# ============================================================

saved = np.load(output_npz)

assert saved["train_angles"].shape == (
    len(train_indices), N_COMPONENTS
)

assert saved["val_angles"].shape == (
    len(val_indices), N_COMPONENTS
)

assert saved["test_angles"].shape == (
    len(test_indices), N_COMPONENTS
)

assert np.isfinite(saved["train_angles"]).all()
assert np.isfinite(saved["val_angles"]).all()
assert np.isfinite(saved["test_angles"]).all()

assert np.array_equal(
    saved["train_original_indices"], train_indices
)

assert np.array_equal(
    saved["val_original_indices"], val_indices
)

assert np.array_equal(
    saved["test_original_indices"], test_indices
)


# ============================================================
# 11. Display results
# ============================================================

print("\n" + "=" * 60)
print("28x28 QUANTUM DATA PREPARATION COMPLETED")
print("=" * 60)

print(f"Image dimensions:    28 x 28")
print(f"Flattened features:  {28 * 28}")
print(f"PCA components:      {N_COMPONENTS}")
print(f"Quantum qubits:      {N_QUBITS}")
print(f"Training angles:     {train_angles.shape}")
print(f"Validation angles:   {val_angles.shape}")
print(f"Test angles:         {test_angles.shape}")
print(f"Excluded samples:    {len(excluded_indices)}")

print(f"\nDataset saved to:\n{output_npz}")
print(f"\nTransformers saved to:\n{transformer_path}")

print("=" * 60)
