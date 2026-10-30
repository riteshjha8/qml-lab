
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# 1. File paths and configuration
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

print("Loading 28x28 dataset...")

if not DATA_28_PATH.exists():
    raise FileNotFoundError(
        f"28x28 dataset not found: {DATA_28_PATH}"
    )

if not DATA_16_PATH.exists():
    raise FileNotFoundError(
        f"Original 16x16 dataset not found: {DATA_16_PATH}"
    )

if not OLD_QUANTUM_PATH.exists():
    raise FileNotFoundError(
        f"Original quantum split not found: {OLD_QUANTUM_PATH}"
    )

data_28 = np.load(DATA_28_PATH)
data_16 = np.load(DATA_16_PATH)
old_quantum = np.load(OLD_QUANTUM_PATH)

X = data_28["X"].astype(np.float32)
y = data_28["y"].astype(np.int64)

old_y = data_16["y"].astype(np.int64)

print(f"28x28 image shape: {X.shape}")
print(f"Labels shape: {y.shape}")


# --------------------------------------------------
# 3. Validate images and labels
# --------------------------------------------------

if X.ndim != 3 or X.shape[1:] != (28, 28):
    raise ValueError(
        f"Expected images with shape (N, 28, 28), got {X.shape}"
    )

if len(X) != len(y):
    raise ValueError("Number of images and labels does not match.")

if not np.isfinite(X).all():
    raise ValueError("Dataset contains NaN or infinite pixel values.")

if X.min() < 0.0 or X.max() > 1.0:
    raise ValueError("Expected normalized pixels in the range [0, 1].")

# Resizing must preserve the image order and labels.
if not np.array_equal(y, old_y):
    raise ValueError(
        "Labels or image order differ between the 16x16 and "
        "28x28 datasets. Do not continue until the ordering is fixed."
    )

print("Dataset validation passed.")


# --------------------------------------------------
# 4. Reuse the ORIGINAL saved split
# --------------------------------------------------

# Important:
# Do not recalculate blank-image indices from resized images.
# Reuse the exact exclusions and train/test indices saved earlier.

excluded_indices = np.asarray(
    old_quantum["excluded_blank_indices"], dtype=np.int64
)

original_train_indices = np.asarray(
    old_quantum["train_original_indices"], dtype=np.int64
)

test_indices = np.asarray(
    old_quantum["test_original_indices"], dtype=np.int64
)

all_indices = np.arange(len(y), dtype=np.int64)

# Validate the saved indices.
for name, indices in [
    ("excluded", excluded_indices),
    ("training", original_train_indices),
    ("test", test_indices),
]:
    if np.any(indices < 0) or np.any(indices >= len(y)):
        raise ValueError(f"{name} indices contain invalid positions.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} indices contain duplicates.")

if np.intersect1d(
    original_train_indices, test_indices
).size:
    raise ValueError("Original train and test sets overlap.")

if np.intersect1d(
    excluded_indices, original_train_indices
).size:
    raise ValueError("Excluded indices overlap with training indices.")

if np.intersect1d(
    excluded_indices, test_indices
).size:
    raise ValueError("Excluded indices overlap with test indices.")

valid_indices = np.setdiff1d(all_indices, excluded_indices)

saved_indices = np.sort(
    np.concatenate([original_train_indices, test_indices])
)

if not np.array_equal(saved_indices, valid_indices):
    raise ValueError(
        "The saved train/test split does not cover exactly the "
        "non-excluded indices. Check the original quantum dataset "
        "before proceeding."
    )

print(f"Excluded indices reused: {excluded_indices.tolist()}")
print(f"Original training samples: {len(original_train_indices)}")
print(f"Test samples: {len(test_indices)}")


# --------------------------------------------------
# 5. Create validation split from training only
# --------------------------------------------------

train_indices, val_indices = train_test_split(
    original_train_indices,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y[original_train_indices],
)

if np.intersect1d(train_indices, val_indices).size:
    raise ValueError("Training and validation sets overlap.")

if np.intersect1d(train_indices, test_indices).size:
    raise ValueError("Training and test sets overlap.")

if np.intersect1d(val_indices, test_indices).size:
    raise ValueError("Validation and test sets overlap.")

print(f"Final training samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")
print(f"Final test samples: {len(test_indices)}")


# --------------------------------------------------
# 6. Flatten 28x28 images
# --------------------------------------------------

X_flat = X.reshape(len(X), 28 * 28)

X_train = X_flat[train_indices]
X_val = X_flat[val_indices]
X_test = X_flat[test_indices]

y_train = y[train_indices]
y_val = y[val_indices]
y_test = y[test_indices]


# --------------------------------------------------
# 7. Fit PCA on training data ONLY
# --------------------------------------------------

print("\nFitting PCA using training data only...")

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE,
)

train_pca = pca.fit_transform(X_train)
val_pca = pca.transform(X_val)
test_pca = pca.transform(X_test)

explained_variance = float(
    pca.explained_variance_ratio_.sum()
)

print(
    f"Explained variance retained by {N_COMPONENTS} components: "
    f"{explained_variance:.4f} ({explained_variance * 100:.2f}%)"
)


# --------------------------------------------------
# 8. Convert PCA features to quantum rotation angles
# --------------------------------------------------

# Fit the scaler on training features only to avoid data leakage.
scaler = MinMaxScaler(feature_range=(0.0, np.pi)) # type: ignore

train_angles = scaler.fit_transform(train_pca).astype(np.float32)
val_angles = scaler.transform(val_pca).astype(np.float32)
test_angles = scaler.transform(test_pca).astype(np.float32)


# --------------------------------------------------
# 9. Save quantum features and split indices
# --------------------------------------------------

output_npz = OUTPUT_DIR / "quantum_features_28.npz"

np.savez_compressed(
    output_npz,
    train_angles=train_angles,
    val_angles=val_angles,
    test_angles=test_angles,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    train_original_indices=train_indices,
    val_original_indices=val_indices,
    test_original_indices=test_indices,
    excluded_blank_indices=excluded_indices,
    explained_variance_ratio=pca.explained_variance_ratio_.astype(
        np.float32
    ),
    explained_variance_total=np.float32(explained_variance),
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


# --------------------------------------------------
# 10. Final verification
# --------------------------------------------------

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

print("\n" + "=" * 55)
print("28x28 QUANTUM PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 55)
print(f"Training angles:   {saved['train_angles'].shape}")
print(f"Validation angles: {saved['val_angles'].shape}")
print(f"Test angles:       {saved['test_angles'].shape}")
print(f"Number of qubits:  {N_QUBITS}")
print(f"PCA components:    {N_COMPONENTS}")
print(f"Output dataset:    {output_npz}")
print(f"Transformers:      {transformer_path}")
print("=" * 55)
