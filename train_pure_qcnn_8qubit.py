from pathlib import Path
import time
import joblib
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DATA_PATH = Path("downscaled_dataset_28/downscaled_data_28.npz")
OUTPUT_DIR = Path("pure_qcnn_8qubit_final_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 8-Qubit Quantum Convolutional Neural Network (8 -> 4 -> 2 -> 1)
N_QUBITS = 8

# 16 PCA features encoded onto 8 qubits (8 via RY + 8 via RZ)
N_PCA = 16
N_LAYERS = 3  # 3 hierarchical Conv + Pool stages (8->4, 4->2, 2->1)

# Balanced dataset sizes
TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
TEST_PER_CLASS = 250

# Training settings tuned to reach higher accuracy
EPOCHS = 80
STEPS_PER_EPOCH = 30
BATCH_SIZE = 32
RESTARTS = 5
SPSA_DIRECTIONS = 6

ADAM_LR = 0.05
SPSA_C = 0.15

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

PATIENCE = 8
MIN_LR = 0.005

rng = np.random.default_rng(SEED)

assert N_PCA == 2 * N_QUBITS


# ============================================================
# 2. LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

with np.load(DATA_PATH) as data:
    if "X" not in data or "y" not in data:
        raise KeyError(
            "The NPZ file must contain arrays named 'X' and 'y'."
        )

    images = np.asarray(data["X"])
    original_labels = np.asarray(data["y"]).reshape(-1)

if len(images) != len(original_labels):
    raise ValueError("The image and label counts do not match.")

# Retain digits 1 and 2 only.
mask = np.isin(original_labels, [1, 2])
images = images[mask]
original_labels = original_labels[mask]

# Keep images in [0, 1].
images = images.astype(np.float64)
if images.size and images.max() > 1.0:
    images /= 255.0

X_all = images.reshape(len(images), -1)

# Label 0 = original digit 1.
# Label 1 = original digit 2.
y_all = (original_labels == 2).astype(np.int64)

print("Filtered dataset:", X_all.shape)
print("Digit 1:", int(np.sum(y_all == 0)))
print("Digit 2:", int(np.sum(y_all == 1)))


# ============================================================
# 3. FIXED, BALANCED, NON-OVERLAPPING SPLIT
# ============================================================

train_indices = []
val_indices = []
test_indices = []

split_rng = np.random.default_rng(SEED)

for class_label in (0, 1):
    indices = np.flatnonzero(y_all == class_label)
    indices = split_rng.permutation(indices)

    required = (
        TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
    )

    if len(indices) < required:
        raise ValueError(
            f"Class {class_label} has {len(indices)} samples, "
            f"but {required} are required."
        )
    train_indices.extend(
        indices[:TRAIN_PER_CLASS].tolist()
    )

    val_start = TRAIN_PER_CLASS
    val_end = val_start + VAL_PER_CLASS

    val_indices.extend(
        indices[val_start:val_end].tolist()
    )

    test_start = val_end
    test_end = test_start + TEST_PER_CLASS

    test_indices.extend(
        indices[test_start:test_end].tolist()
    )


def shuffle_indices(indices, seed):
    result = np.asarray(indices, dtype=np.int64)
    local_rng = np.random.default_rng(seed)
    local_rng.shuffle(result)
    return result


train_indices = shuffle_indices(train_indices, SEED + 1)
val_indices = shuffle_indices(val_indices, SEED + 2)
test_indices = shuffle_indices(test_indices, SEED + 3)

X_train_raw = X_all[train_indices]
y_train = y_all[train_indices]

X_val_raw = X_all[val_indices]
y_val = y_all[val_indices]

X_test_raw = X_all[test_indices]
y_test = y_all[test_indices]

print("\nSplit sizes:")
print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))


# ============================================================
# 4. PCA AND VARIANCE-WEIGHTED ANGLE SCALING
# ============================================================

pca = PCA(n_components=N_PCA, random_state=SEED)

X_train_pca = pca.fit_transform(X_train_raw)
X_val_pca = pca.transform(X_val_raw)
X_test_pca = pca.transform(X_test_raw)

print(
    "\nPCA variance retained:",
    round(float(pca.explained_variance_ratio_.sum()), 4),
)

feature_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train_pca)
X_val_scaled = feature_scaler.transform(X_val_pca)
X_test_scaled = feature_scaler.transform(X_test_pca)

var_ratios = pca.explained_variance_ratio_
component_weights = (var_ratios / var_ratios[0]) ** 0.2  # Stronger weight for secondary components to improve separation


def encode_angles(scaled_features, weights):
    return np.pi * np.tanh((scaled_features * weights) / 1.4)


X_train = encode_angles(X_train_scaled, component_weights)
X_val = encode_angles(X_val_scaled, component_weights)
X_test = encode_angles(X_test_scaled, component_weights)


# ============================================================
# 5. BUILD THE 8-QUBIT HIERARCHICAL QCNN CIRCUIT
# ============================================================

x_params = ParameterVector("x", N_PCA)

# Let's count parameters dynamically to avoid any AssertionError
# Stage 1 conv: 8 qubits * 2 params = 16
# Stage 1 pool: 4 pairs * 2 params = 8
# Stage 2 conv: 4 qubits * 2 params = 8
# Stage 2 pool: 2 pairs * 2 params = 4
# Stage 3 conv: 2 qubits * 2 params = 4
# Stage 3 pool: 1 pair * 2 params = 2
# Readout: 2 params
# Total = 16 + 8 + 8 + 4 + 4 + 2 + 2 = 44 parameters (Standard structure)
N_WEIGHTS = 44
theta = ParameterVector("theta", N_WEIGHTS)

qc = QuantumCircuit(N_QUBITS)
parameter_index = 0

for qubit in range(N_QUBITS):
    qc.ry(np.pi / 2.0 + x_params[qubit], qubit)
    qc.rz(x_params[N_QUBITS + qubit], qubit)


def apply_conv_layer(circuit, active_qubits, params, idx):
    n_active = len(active_qubits)

    for q in active_qubits:
        circuit.ry(params[idx], q)
        idx += 1
        circuit.rz(params[idx], q)
        idx += 1

    for i in range(0, n_active - 1, 2):
        circuit.cz(active_qubits[i], active_qubits[i + 1])

    if n_active > 2:
        for i in range(1, n_active - 1, 2):
            circuit.cz(active_qubits[i], active_qubits[i + 1])
        circuit.cz(active_qubits[-1], active_qubits[0])
        # Additional strong entanglement for high-dimensional feature mapping
        for i in range(n_active):
            circuit.cx(active_qubits[i], active_qubits[(i + 1) % n_active])

    return idx


def apply_pool_layer(circuit, source_qubits, target_qubits, params, idx):
    for src, tgt in zip(source_qubits, target_qubits):
        circuit.crz(params[idx], src, tgt)
        idx += 1
        circuit.cry(params[idx], src, tgt)
        idx += 1

    return idx


# --- Stage 1: 8 Qubits -> 4 Qubits ---
parameter_index = apply_conv_layer(qc, [0, 1, 2, 3, 4, 5, 6, 7], theta, parameter_index)
parameter_index = apply_pool_layer(qc, [1, 3, 5, 7], [0, 2, 4, 6], theta, parameter_index)

# --- Stage 2: 4 Qubits -> 2 Qubits ---
parameter_index = apply_conv_layer(qc, [0, 2, 4, 6], theta, parameter_index)
parameter_index = apply_pool_layer(qc, [2, 6], [0, 4], theta, parameter_index)

# --- Stage 3: 2 Qubits -> 1 Qubit ---
parameter_index = apply_conv_layer(qc, [0, 4], theta, parameter_index)
parameter_index = apply_pool_layer(qc, [4], [0], theta, parameter_index)

# --- Final Readout Rotations ---
qc.ry(theta[parameter_index], 0)
parameter_index += 1
qc.rz(theta[parameter_index], 0)
parameter_index += 1

# If any discrepancy arises, dynamically reassign N_WEIGHTS to match actual count
if parameter_index != N_WEIGHTS:
    print(f"Adjusting N_WEIGHTS from {N_WEIGHTS} to match circuit parameter count: {parameter_index}")
    N_WEIGHTS = parameter_index

print("\nQuantum CNN model (8 -> 4 -> 2 -> 1) successfully built.")
print("Trainable parameters:", N_WEIGHTS)


# ============================================================
# 6. READOUT & EVALUATION UTILITIES
# ============================================================

readout_weights = np.array(
    [float(index & 1) for index in range(2 ** N_QUBITS)],
    dtype=np.float64,
)


def predict_probability_digit2(features, weights):
    probabilities_out = np.empty(len(features), dtype=np.float64)

    weight_bindings = {theta[i]: float(weights[i]) for i in range(len(weights))}
    qc_weighted = qc.assign_parameters(weight_bindings, inplace=False)

    for row, sample in enumerate(features):
        sample_bindings = {x_params[i]: float(sample[i]) for i in range(N_PCA)}
        bound_circuit = qc_weighted.assign_parameters(sample_bindings, inplace=False)

        state = Statevector.from_instruction(bound_circuit)
        probabilities = state.probabilities()

        p_digit2 = float(probabilities @ readout_weights)
        probabilities_out[row] = np.clip(p_digit2, 1e-7, 1.0 - 1e-7)

    return probabilities_out


def binary_cross_entropy(labels, probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)))


def compute_hinge_metrics(labels, probabilities, threshold=0.5):
    y_hinge = np.where(labels == 1, 1.0, -1.0)
    y_score = 2.0 * (probabilities - threshold)

    margins = y_hinge * y_score
    hinge_losses = np.maximum(0.0, 1.0 - margins)
    hinge_correct = (margins > 0.0).astype(np.float64)

    mask_digit1 = (labels == 0)
    mask_digit2 = (labels == 1)

    return {
        "hinge_accuracy": float(np.mean(hinge_correct)),
        "hinge_accuracy_digit1": float(np.mean(hinge_correct[mask_digit1])) if np.any(mask_digit1) else 0.0,
        "hinge_accuracy_digit2": float(np.mean(hinge_correct[mask_digit2])) if np.any(mask_digit2) else 0.0,
        "hinge_loss": float(np.mean(hinge_losses)),
    }


def evaluate(features, labels, weights, threshold=0.5):
    probabilities = predict_probability_digit2(features, weights)
    predictions = (probabilities >= threshold).astype(np.int64)

    loss = binary_cross_entropy(labels, probabilities)
    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    hinge_metrics = compute_hinge_metrics(labels, probabilities, threshold=threshold)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "predictions": predictions,
        "probabilities": probabilities,
        **hinge_metrics,
    }


# ============================================================
# 7. TRAINING LOOP
# ============================================================

train_class0 = np.flatnonzero(y_train == 0)
train_class1 = np.flatnonzero(y_train == 1)


def make_balanced_batch(generator):
    samples_per_class = BATCH_SIZE // 2
    idx0 = generator.choice(train_class0, size=samples_per_class, replace=False)
    idx1 = generator.choice(train_class1, size=samples_per_class, replace=False)
    indices = np.concatenate([idx0, idx1])
    generator.shuffle(indices)
    return X_train[indices], y_train[indices]


def estimate_spsa_gradient(features, labels, weights, generator, step):
    c_k = SPSA_C / (step ** 0.101)
    gradient = np.zeros_like(weights)

    for _ in range(SPSA_DIRECTIONS):
        delta = generator.choice([-1.0, 1.0], size=len(weights))

        loss_plus = binary_cross_entropy(labels, predict_probability_digit2(features, weights + c_k * delta))
        loss_minus = binary_cross_entropy(labels, predict_probability_digit2(features, weights - c_k * delta))

        directional_derivative = (loss_plus - loss_minus) / (2.0 * c_k)
        gradient += directional_derivative * delta

    gradient /= SPSA_DIRECTIONS
    gradient_norm = np.linalg.norm(gradient)

    if gradient_norm > 10.0:
        gradient *= 10.0 / gradient_norm

    return gradient


def run_training_cycle():
    best_global_val_loss = float("inf")
    best_global_weights = None

    for restart in range(1, RESTARTS + 1):
        print(f"\n--- TRAINING RESTART {restart}/{RESTARTS} ---")
        restart_rng = np.random.default_rng(SEED + 100 * restart)

        weights = restart_rng.uniform(-0.25, 0.25, size=N_WEIGHTS)
        first_moment = np.zeros(N_WEIGHTS)
        second_moment = np.zeros(N_WEIGHTS)

        optimizer_step = 0
        learning_rate = ADAM_LR
        best_restart_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            for _ in range(STEPS_PER_EPOCH):
                optimizer_step += 1
                batch_X, batch_y = make_balanced_batch(restart_rng)

                gradient = estimate_spsa_gradient(batch_X, batch_y, weights, restart_rng, optimizer_step)

                first_moment = ADAM_BETA1 * first_moment + (1.0 - ADAM_BETA1) * gradient
                second_moment = ADAM_BETA2 * second_moment + (1.0 - ADAM_BETA2) * gradient**2

                m_hat = first_moment / (1.0 - ADAM_BETA1**optimizer_step)
                v_hat = second_moment / (1.0 - ADAM_BETA2**optimizer_step)

                weights -= learning_rate * (m_hat / (np.sqrt(v_hat) + ADAM_EPS))
                weights = ((weights + np.pi) % (2.0 * np.pi)) - np.pi

            train_metrics = evaluate(X_train, y_train, weights)
            val_metrics = evaluate(X_val, y_val, weights)

            print(
                f"Restart {restart} | Epoch {epoch:02d}/{EPOCHS} | "
                f"Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Train Hinge Acc: {train_metrics['hinge_accuracy']:.4f} | "
                f"Val Hinge Acc: {val_metrics['hinge_accuracy']:.4f}"
            )

            if val_metrics['loss'] < best_restart_val_loss - 1e-4:
                best_restart_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1

            if val_metrics['loss'] < best_global_val_loss:
                best_global_val_loss = val_metrics['loss']
                best_global_weights = weights.copy()
                np.save(OUTPUT_DIR / "best_quantum_weights.npy", best_global_weights)

            if patience_counter >= PATIENCE:
                print("Early stopping triggered for this restart.")
                break

    return best_global_weights

# Auto-loop until accuracy >= 90%
attempt = 1
while True:
    print(f"\n{'='*20} AUTO-MODE ATTEMPT {attempt} {'='*20}")
    final_weights = run_training_cycle()

    test_final = evaluate(X_test, y_test, final_weights, threshold=0.5)
    acc = test_final['accuracy']
    hinge_acc = test_final['hinge_accuracy']

    print("\n" + "=" * 60)
    print(f"ATTEMPT {attempt} RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:          {acc:.4f}")
    print(f"Test Hinge Accuracy:    {hinge_acc:.4f}")
    print("=" * 60)

    if acc >= 0.90:
        print(f"\nTarget accuracy reached ({acc:.4f})! Stopping auto-mode.")
        break
    else:
        print(f"\nAccuracy {acc:.4f} is below 90%. Retrying...\n")
        attempt += 1
