
from pathlib import Path
import time
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector


# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
DIGITS = [0, 1, 2, 9]
NUM_CLASSES = len(DIGITS)
NUM_QUBITS = 4

TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
TEST_PER_CLASS = 250

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.01
PATIENCE = 4

DATA_PATH = Path("downscaled_dataset_28/downscaled_data_28.npz")
OUT_DIR = Path("qcnn_results_4class")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Device:", device)


# ============================================================
# 1. LOAD EXISTING 28x28 DATA
# ============================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

data = np.load(DATA_PATH)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

if X.ndim != 3 or X.shape[1:] != (28, 28):
    raise ValueError(
        "Expected X with shape (N, 28, 28). "
        "Check the selected NPZ file."
    )

# Normalize if the source data is stored as 0..255.
if X.max() > 1.0:
    X /= 255.0

# Filter to digits 0, 1, 2, and 9.
mask = np.isin(y, DIGITS)
X = X[mask]
y = y[mask]

# Remap original labels:
# digit 0 -> class 0
# digit 1 -> class 1
# digit 2 -> class 2
# digit 9 -> class 3
label_map = {digit: i for i, digit in enumerate(DIGITS)}
y = np.array([label_map[int(label)] for label in y],
             dtype=np.int64)

print("\nFiltered digit counts:")
for digit, class_id in label_map.items():
    print(f"Digit {digit}: {np.sum(y == class_id)}")

# Split each digit independently so that all three sets
# have exactly balanced class counts.
rng = np.random.default_rng(SEED)
train_indices, val_indices, test_indices = [], [], []

for class_id in range(NUM_CLASSES):
    indices = np.flatnonzero(y == class_id)
    rng.shuffle(indices)

    required = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
    if len(indices) < required:
        raise ValueError(
            f"Class digit {DIGITS[class_id]} has {len(indices)} "
            f"images, but {required} are needed. "
            "Reduce the per-class sample settings."
        )

    train_indices.extend(indices[:TRAIN_PER_CLASS])
    val_start = TRAIN_PER_CLASS
    test_start = TRAIN_PER_CLASS + VAL_PER_CLASS

    val_indices.extend(indices[val_start:test_start])
    test_indices.extend(indices[test_start:required])

train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
test_indices = np.array(test_indices)

# Shuffle each partition without mixing the partitions.
rng.shuffle(train_indices)
rng.shuffle(val_indices)
rng.shuffle(test_indices)

X_train_img, y_train = X[train_indices], y[train_indices]
X_val_img, y_val = X[val_indices], y[val_indices]
X_test_img, y_test = X[test_indices], y[test_indices]

print("\nSplit sizes:")
print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))

assert len(y_train) == 2000
assert len(y_val) == 400
assert len(y_test) == 1000


# ============================================================
# 2. CONVERT 28x28 IMAGES TO FOUR QUANTUM FEATURES
# ============================================================
# Flatten each image. Fit PCA only on training images.
X_train_flat = X_train_img.reshape(len(X_train_img), -1)
X_val_flat = X_val_img.reshape(len(X_val_img), -1)
X_test_flat = X_test_img.reshape(len(X_test_img), -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled = scaler.transform(X_val_flat)
X_test_scaled = scaler.transform(X_test_flat)

pca = PCA(n_components=NUM_QUBITS, random_state=SEED)
X_train_features = pca.fit_transform(X_train_scaled)
X_val_features = pca.transform(X_val_scaled)
X_test_features = pca.transform(X_test_scaled)

# Scale features into a suitable rotation-angle range.
feature_min = X_train_features.min(axis=0)
feature_max = X_train_features.max(axis=0)
denominator = np.maximum(feature_max - feature_min, 1e-8)

def scale_angles(features):
    scaled = 2.0 * (features - feature_min) / denominator - 1.0
    return (scaled * np.pi).astype(np.float32)

X_train_features = scale_angles(X_train_features)
X_val_features = scale_angles(X_val_features)
X_test_features = scale_angles(X_test_features)

print("Quantum input shape:", X_train_features.shape)
print("PCA explained variance:",
      round(float(pca.explained_variance_ratio_.sum()), 4))

# Save preprocessing parameters for reproducibility.
np.savez(
    OUT_DIR / "preprocessing.npz",
    scaler_mean=scaler.mean_, # type: ignore
    scaler_scale=scaler.scale_, # type: ignore
    pca_components=pca.components_,
    pca_mean=pca.mean_,
    feature_min=feature_min,
    feature_max=feature_max,
    original_digits=np.array(DIGITS),
)


# ============================================================
# 3. BUILD FOUR-QUBIT VARIATIONAL QUANTUM CIRCUIT
# ============================================================
x_params = ParameterVector("x", NUM_QUBITS)
theta = ParameterVector("theta", NUM_QUBITS * 2)

circuit = QuantumCircuit(NUM_QUBITS)

# Data encoding.
for q in range(NUM_QUBITS):
    circuit.ry(x_params[q], q)

# Entangle neighboring qubits.
for q in range(NUM_QUBITS - 1):
    circuit.cx(q, q + 1)
circuit.cx(NUM_QUBITS - 1, 0)

# Two trainable variational layers.
weight_index = 0
for _ in range(2):
    for q in range(NUM_QUBITS):
        circuit.ry(theta[weight_index], q)
        weight_index += 1

    for q in range(NUM_QUBITS - 1):
        circuit.cx(q, q + 1)
    circuit.cx(NUM_QUBITS - 1, 0)

print("\nQuantum circuit depth:", circuit.depth())
print("Trainable quantum parameters:", len(theta))


# Map computational-basis outcomes into four output classes.
# Outcomes are grouped by their two least-significant bits.
def interpret_outcome(bitstring):
    return int(bitstring) % NUM_CLASSES


sampler = StatevectorSampler(seed=SEED)

qnn = SamplerQNN(
    circuit=circuit,
    sampler=sampler,
    input_params=list(x_params),
    weight_params=list(theta),
    interpret=interpret_outcome,
    output_shape=NUM_CLASSES,
    input_gradients=False,
)

initial_weights = 0.1 * (
    2 * torch.rand(len(theta), dtype=torch.float32) - 1
)

quantum_layer = TorchConnector(
    qnn,
    initial_weights=initial_weights,
)


# ============================================================
# 4. HYBRID MODEL
# ============================================================
class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.classical = nn.Sequential(
            nn.Linear(NUM_QUBITS, 8),
            nn.ReLU(),
            nn.Linear(8, NUM_CLASSES),
        )

        self.quantum = quantum_layer

        # Combine classical and quantum scores.
        self.output = nn.Linear(NUM_CLASSES * 2, NUM_CLASSES)

    def forward(self, features):
        classical_scores = self.classical(features)
        quantum_probs = self.quantum(features)

        combined = torch.cat(
            [classical_scores, quantum_probs], dim=1
        )
        return self.output(combined)


model = HybridQNN().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)
criterion = nn.CrossEntropyLoss()

print("Classical + quantum parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================================================
# 5. DATA LOADERS
# ============================================================
def make_loader(features, labels, shuffle):
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = make_loader(X_train_features, y_train, True)
val_loader = make_loader(X_val_features, y_val, False)
test_loader = make_loader(X_test_features, y_test, False)


# ============================================================
# 6. TRAINING WITH VALIDATION CHECKPOINTING
# ============================================================
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            predictions = outputs.argmax(dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())

    return (
        total_loss / len(all_true),
        accuracy_score(all_true, all_pred),
        f1_score(
            all_true, all_pred,
            average="macro",
            zero_division=0,
        ),
        np.array(all_true),
        np.array(all_pred),
    )


best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0
history = []
start_time = time.time()

print("\nStarting training...")
print("Quantum simulations can be slow on CPU.")

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()

    running_loss = 0.0
    train_true, train_pred = [], []

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(labels)
        train_true.extend(labels.cpu().numpy())
        train_pred.extend(outputs.argmax(dim=1).detach().cpu().numpy())

    train_loss = running_loss / len(train_true)
    train_acc = accuracy_score(train_true, train_pred)
    train_f1 = f1_score(
        train_true, train_pred, average="macro", zero_division=0
    )

    val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss={train_loss:.4f} "
        f"acc={train_acc:.4f} f1={train_f1:.4f} | "
        f"val loss={val_loss:.4f} "
        f"acc={val_acc:.4f} f1={val_f1:.4f} | "
        f"time={time.time() - epoch_start:.1f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "model_state_dict": best_state,
                "digits": DIGITS,
                "num_qubits": NUM_QUBITS,
                "best_val_loss": best_val_loss,
            },
            OUT_DIR / "best_model.pt",
        )
        epochs_without_improvement = 0
        print("  Best checkpoint saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation loss did not improve.")
        break


# ============================================================
# 7. FINAL TEST ON EXACTLY 1000 HELD-OUT IMAGES
# ============================================================
if best_state is None:
    raise RuntimeError("No model checkpoint was saved.")

model.load_state_dict(best_state)

test_loss, test_acc, test_f1, test_true, test_pred = evaluate(test_loader)

print("\n" + "=" * 60)
print("FINAL FOUR-CLASS TEST RESULTS")
print("=" * 60)
print("Original digits:", DIGITS)
print("Test images:", len(test_true))
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4%}")
print(f"Test macro F1: {test_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        test_true,
        test_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[str(d) for d in DIGITS],
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(test_true, test_pred))

np.savez(
    OUT_DIR / "test_predictions.npz",
    y_true=test_true,
    y_pred=test_pred,
    test_loss=np.array(test_loss),
    test_accuracy=np.array(test_acc),
    test_macro_f1=np.array(test_f1),
    original_digits=np.array(DIGITS),
)

import csv
with open(OUT_DIR / "training_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

print(f"\nResults saved to: {OUT_DIR.resolve()}")
print(f"Total elapsed time: {(time.time() - start_time) / 60:.1f} minutes")
