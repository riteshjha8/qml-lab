from pathlib import Path
import time
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
DIGITS = [1, 2]

DATA_PATH = Path("downscaled_dataset_28/downscaled_data_28.npz")
OUTPUT_DIR = Path("qnn_2class_v2_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 4
N_Q_LAYERS = 2

# CHANGE 1: Keep 16 PCA features for the classical branch.
N_PCA = 16

# Quantum branch still uses only four features and four qubits.
N_Q_FEATURES = 4

TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
TEST_PER_CLASS = 250

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.004
PATIENCE = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
print("Device:", DEVICE)


# ============================================================
# 2. LOAD EXISTING DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

data = np.load(DATA_PATH)
X_all = data["X"]
y_all = data["y"]

print("Original images:", X_all.shape)
print("Original labels:", y_all.shape)

# Keep only digits 1 and 2.
mask = np.isin(y_all, DIGITS)
X = X_all[mask].astype(np.float32)
y_original = y_all[mask].astype(np.int64)

# Remap digit 1 -> 0 and digit 2 -> 1.
y = (y_original == 2).astype(np.int64)

print("\nSelected digit counts:")
for digit in DIGITS:
    print(f"Digit {digit}: {np.sum(y_original == digit)}")

# Flatten 28x28 images for StandardScaler and PCA.
X = X.reshape(len(X), -1)


# ============================================================
# 3. BALANCED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

test_indices = []
remaining_indices = []

for class_label in [0, 1]:
    indices = np.where(y == class_label)[0].copy()
    np.random.shuffle(indices)

    needed = (
        TRAIN_PER_CLASS
        + VAL_PER_CLASS
        + TEST_PER_CLASS
    )

    if len(indices) < needed:
        raise ValueError(
            f"Class {class_label} has {len(indices)} images, "
            f"but {needed} are required. Reduce split sizes."
        )

    test_indices.extend(indices[:TEST_PER_CLASS])
    remaining_indices.extend(indices[TEST_PER_CLASS:])

test_indices = np.array(test_indices)
remaining_indices = np.array(remaining_indices)

train_indices = []
val_indices = []

for class_label in [0, 1]:
    indices = remaining_indices[
        y[remaining_indices] == class_label
    ].copy()

    np.random.shuffle(indices)

    train_indices.extend(indices[:TRAIN_PER_CLASS])

    val_start = TRAIN_PER_CLASS
    val_end = TRAIN_PER_CLASS + VAL_PER_CLASS

    val_indices.extend(indices[val_start:val_end])

train_indices = np.array(train_indices)
val_indices = np.array(val_indices)

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)
np.random.shuffle(test_indices)

X_train_raw = X[train_indices]
y_train = y[train_indices]

X_val_raw = X[val_indices]
y_val = y[val_indices]

X_test_raw = X[test_indices]
y_test = y[test_indices]

print("\nSplit sizes:")
print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))


# ============================================================
# 4. SCALE AND REDUCE TO 16 PCA FEATURES
# ============================================================

# Fit preprocessing using training data only.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(
    n_components=N_PCA,
    random_state=SEED,
)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# CHANGE 2: Classical branch uses all 16 PCA features.
X_train_classical = X_train_pca.astype(np.float32)
X_val_classical = X_val_pca.astype(np.float32)
X_test_classical = X_test_pca.astype(np.float32)

# Quantum branch uses only the first four PCA features.
X_train_q = X_train_pca[:, :N_Q_FEATURES].copy()
X_val_q = X_val_pca[:, :N_Q_FEATURES].copy()
X_test_q = X_test_pca[:, :N_Q_FEATURES].copy()

# Convert quantum features to rotation angles.
# Use training data only to determine the scale.
angle_scale = float(np.max(np.abs(X_train_q)))

if angle_scale == 0:
    angle_scale = 1.0

X_train_q = (
    X_train_q / angle_scale * np.pi
).astype(np.float32)

X_val_q = (
    X_val_q / angle_scale * np.pi
).astype(np.float32)

X_test_q = (
    X_test_q / angle_scale * np.pi
).astype(np.float32)

print(
    "\nPCA variance retained:",
    round(
        float(np.sum(pca.explained_variance_ratio_)),
        4,
    ),
)

print("Classical features:", X_train_classical.shape[1])
print("Quantum features:", X_train_q.shape[1])


# ============================================================
# 5. BUILD TWO-CLASS QUANTUM CIRCUIT
# ============================================================

def create_qnn():
    num_weights = N_QUBITS * N_Q_LAYERS

    circuit = QuantumCircuit(N_QUBITS)

    inputs = ParameterVector("x", N_Q_FEATURES)
    weights = ParameterVector("theta", num_weights)

    # Encode four input features with Ry rotations.
    for qubit in range(N_QUBITS):
        circuit.ry(inputs[qubit], qubit)

    weight_index = 0

    for layer in range(N_Q_LAYERS):
        # Entangle neighboring qubits.
        for qubit in range(N_QUBITS - 1):
            circuit.cx(qubit, qubit + 1)

        # Close the entanglement ring.
        circuit.cx(N_QUBITS - 1, 0)

        # Trainable rotations.
        for qubit in range(N_QUBITS):
            circuit.ry(weights[weight_index], qubit)
            weight_index += 1

    circuit.measure_all()

    # Even measurement parity -> class 0.
    # Odd measurement parity -> class 1.
    def interpret(bitstring):
        return int(bitstring).bit_count() % 2

    sampler = StatevectorSampler(
        default_shots=256,
        seed=SEED,
    )

    qnn = SamplerQNN(
        circuit=circuit,
        sampler=sampler,
        input_params=list(inputs),
        weight_params=list(weights),
        interpret=interpret,
        output_shape=2,
        input_gradients=False,
    )

    initial_weights = (
        0.1 * np.random.randn(num_weights)
    ).astype(np.float32)

    return TorchConnector(
        qnn,
        initial_weights=torch.tensor(
            initial_weights,
            dtype=torch.float32,
        ),
    )


# ============================================================
# 6. HYBRID CLASSICAL + QUANTUM MODEL
# ============================================================

class BinaryHybridQNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Classical branch receives all 16 PCA features.
        self.classical = nn.Sequential(
            nn.Linear(N_PCA, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        # Quantum branch receives four features.
        self.quantum = create_qnn()

        # Combine two classical and two quantum outputs.
        self.classifier = nn.Linear(4, 2)

    # CHANGE 3: Separate classical and quantum inputs.
    def forward(self, classical_x, quantum_x):
        classical_output = self.classical(classical_x)
        quantum_output = self.quantum(quantum_x)

        combined = torch.cat(
            [classical_output, quantum_output],
            dim=1,
        )

        return self.classifier(combined)


# ============================================================
# 7. CREATE DATA LOADERS WITH TWO INPUTS
# ============================================================

def make_loader(classical_features, quantum_features, labels, shuffle):
    dataset = TensorDataset(
        torch.tensor(
            classical_features,
            dtype=torch.float32,
        ),
        torch.tensor(
            quantum_features,
            dtype=torch.float32,
        ),
        torch.tensor(
            labels,
            dtype=torch.long,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = make_loader(
    X_train_classical,
    X_train_q,
    y_train,
    True,
)

val_loader = make_loader(
    X_val_classical,
    X_val_q,
    y_val,
    False,
)

test_loader = make_loader(
    X_test_classical,
    X_test_q,
    y_test,
    False,
)


# ============================================================
# 8. TRAINING AND EVALUATION
# ============================================================

model = BinaryHybridQNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_accuracy = -1.0
best_state = None
best_epoch = 0
epochs_without_improvement = 0
history = []


def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for classical_x, quantum_x, labels in loader:
            classical_x = classical_x.to(DEVICE)
            quantum_x = quantum_x.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(classical_x, quantum_x)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)

            predictions = logits.argmax(dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_true, all_pred)

    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )

    return (
        total_loss / len(loader.dataset),
        accuracy,
        macro_f1,
        np.array(all_true),
        np.array(all_pred),
    )


print("\nStarting updated binary QNN training...")
print("Digit 1 -> label 0")
print("Digit 2 -> label 1")
print("Classical PCA features:", N_PCA)
print("Quantum input features:", N_Q_FEATURES)
print("Learning rate:", LEARNING_RATE)

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for classical_x, quantum_x, labels in train_loader:
        classical_x = classical_x.to(DEVICE)
        quantum_x = quantum_x.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(classical_x, quantum_x)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

        predictions = logits.argmax(dim=1)

        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(predictions.detach().cpu().numpy())

    train_loss = total_loss / len(train_loader.dataset) # type: ignore

    train_accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    train_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )

    (
        val_loss,
        val_accuracy,
        val_f1,
        _,
        _,
    ) = evaluate(model, val_loader)

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss={train_loss:.4f} "
        f"acc={train_accuracy:.4f} "
        f"f1={train_f1:.4f} | "
        f"val loss={val_loss:.4f} "
        f"acc={val_accuracy:.4f} "
        f"f1={val_f1:.4f} | "
        f"time={elapsed:.1f}s"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "seconds": elapsed,
    })

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save(
            best_state,
            OUTPUT_DIR / "best_binary_qnn_v2.pt",
        )

        print("  Best checkpoint saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            "Early stopping: validation accuracy "
            "did not improve."
        )
        break


# ============================================================
# 9. TEST THE BEST MODEL
# ============================================================

if best_state is None:
    raise RuntimeError(
        "Training did not produce a model checkpoint."
    )

model.load_state_dict(best_state)

(
    test_loss,
    test_accuracy,
    test_f1,
    y_true,
    y_pred,
) = evaluate(model, test_loader)

print("\n========== FINAL TEST RESULTS ==========")
print("Best epoch:", best_epoch)
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Test loss:                {test_loss:.4f}")
print(f"Test accuracy:             {test_accuracy:.4f}")
print(f"Test macro F1:             {test_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Digit 1", "Digit 2"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

np.savez(
    OUTPUT_DIR / "test_predictions.npz",
    y_true=y_true,
    y_pred=y_pred,
)

with open(
    OUTPUT_DIR / "training_history.json",
    "w",
) as file:
    json.dump(history, file, indent=2)

print("\nResults saved in:", OUTPUT_DIR.resolve())
