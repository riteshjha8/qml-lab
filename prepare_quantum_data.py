
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------
# STEP 5A: Classical image data -> PCA features -> quantum
# angle encoding, using Qiskit and Qiskit Aer.
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "downscaled_dataset" / "downscaled_data.npz"
OUTPUT_DIR = ROOT / "quantum_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 8
TEST_SIZE = 0.20
RANDOM_STATE = 42
INK_THRESHOLD = 0.15

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

# 1. Load the prepared images and labels.
data = np.load(INPUT_FILE)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

if X.ndim != 3 or X.shape[1:] != (16, 16):
    raise ValueError(f"Expected images shaped (N, 16, 16), got {X.shape}")

if len(X) != len(y):
    raise ValueError("Image and label counts do not match.")

if not np.isfinite(X).all():
    raise ValueError("Input contains NaN or infinite values.")

if X.min() < 0 or X.max() > 1:
    raise ValueError("Expected normalized pixel values between 0 and 1.")

# 2. Exclude completely blank images from the modeling dataset.
# The source NPZ is never modified.
ink_counts = (X > INK_THRESHOLD).sum(axis=(1, 2))
valid_mask = ink_counts > 0

excluded_indices = np.flatnonzero(~valid_mask)
X = X[valid_mask]
y = y[valid_mask]

if len(X) == 0:
    raise ValueError("No nonblank images remain.")

# Flatten 16x16 images into 256-dimensional feature vectors.
X_flat = X.reshape(len(X), -1)

# Keep the original dataset positions for traceability.
original_indices = np.flatnonzero(valid_mask)

# 3. Stratified split: 80% training, 20% testing.
train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train = X_flat[train_idx]
X_test = X_flat[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# 4. Fit PCA on training images only to prevent data leakage.
pca = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 5. Map PCA features to rotation angles.
# The scaler is fitted only on training PCA features.

scaler = MinMaxScaler(feature_range=(0.0,np.pi)) # type: ignore
train_angles = scaler.fit_transform(X_train_pca)
test_angles = scaler.transform(X_test_pca)

# Note: test angles may fall outside [0, pi] if a test feature
# exceeds the training range. This is valid for RY rotations.

# 6. Build an example data-encoding circuit.
# One PCA feature is encoded into each qubit with an RY gate.
example_angles = train_angles[0]

encoding_circuit = QuantumCircuit(N_QUBITS)

for qubit, angle in enumerate(example_angles):
    encoding_circuit.ry(float(angle), qubit)

# Save the circuit diagram as readable text.
diagram_path = OUTPUT_DIR / "example_encoding_circuit.txt"
diagram_path.write_text(
    str(encoding_circuit.draw(output="text")),
    encoding="utf-8",
)

# 7. Simulate the encoding circuit with Qiskit Aer.
# Measurement is added to a copy for a simple execution check.
measured_circuit = encoding_circuit.copy()
measured_circuit.measure_all()

simulator = AerSimulator()
compiled_circuit = transpile(measured_circuit, simulator)
simulation_result = simulator.run(
    compiled_circuit,
    shots=1024,
    seed_simulator=RANDOM_STATE,
).result()

counts = simulation_result.get_counts()

# 8. Save processed features, labels and split indices.
np.savez_compressed(
    OUTPUT_DIR / "quantum_features.npz",
    X_train_pca=X_train_pca,
    X_test_pca=X_test_pca,
    train_angles=train_angles,
    test_angles=test_angles,
    y_train=y_train,
    y_test=y_test,
    train_original_indices=original_indices[train_idx],
    test_original_indices=original_indices[test_idx],
    excluded_blank_indices=excluded_indices,
    explained_variance_ratio=pca.explained_variance_ratio_,
    explained_variance_total=np.array(
        pca.explained_variance_ratio_.sum()
    ),
    n_qubits=np.array(N_QUBITS),
    random_state=np.array(RANDOM_STATE),
)

# Save the PCA and angle-scaling objects for reproducibility.
import joblib

joblib.dump(
    {"pca": pca, "angle_scaler": scaler},
    OUTPUT_DIR / "feature_transformers.joblib",
)

# 9. Print verification results.
print("\nSTEP 5A: QUANTUM DATA PREPARATION COMPLETED")
print("-" * 52)
print(f"Original image count: {len(data['y'])}")
print(f"Blank images excluded: {len(excluded_indices)}")
print(f"Images retained: {len(y)}")
print(f"Training samples: {len(y_train)}")
print(f"Testing samples: {len(y_test)}")
print(f"Features before PCA: {X_flat.shape[1]}")
print(f"Features after PCA: {X_train_pca.shape[1]}")
print(f"Qubits in example circuit: {N_QUBITS}")
print(
    "Variance explained by 8 PCA components: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)
print(f"Training angles shape: {train_angles.shape}")
print(f"Testing angles shape: {test_angles.shape}")
print(f"Encoding circuit diagram: {diagram_path}")
print(f"Simulation measurement shots: {sum(counts.values())}")
print(f"Simulation measurement outcomes: {counts}")
print(f"Saved output directory: {OUTPUT_DIR}")
print("\nNo source dataset files were modified.")
