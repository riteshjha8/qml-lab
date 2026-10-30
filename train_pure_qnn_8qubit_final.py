from pathlib import Path
import time

import joblib
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from sklearn.decomposition import PCA
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

# Balanced dataset sizes, matching the previous experiment
TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
TEST_PER_CLASS = 250

# Training settings
EPOCHS = 30
STEPS_PER_EPOCH = 10
BATCH_SIZE = 32
RESTARTS = 3
SPSA_DIRECTIONS = 3

ADAM_LR = 0.008
SPSA_C = 0.15

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

PATIENCE = 8
MIN_LR = 0.003

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

# Select the same number of examples from each class.
# Test and validation samples are not used for fitting PCA.

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

print(
    "Train class counts:",
    np.bincount(y_train, minlength=2),
)
print(
    "Validation class counts:",
    np.bincount(y_val, minlength=2),
)
print(
    "Test class counts:",
    np.bincount(y_test, minlength=2),
)


# ============================================================
# 4. PCA AND VARIANCE-PRESERVING ANGLE SCALING (UPDATED)
# ============================================================

pca = PCA(n_components=N_PCA, random_state=SEED)

X_train_pca = pca.fit_transform(X_train_raw)
X_val_pca = pca.transform(X_val_raw)
X_test_pca = pca.transform(X_test_raw)

print(
    "\nPCA variance retained:",
    round(float(pca.explained_variance_ratio_.sum()), 4),
)

# Preserve relative PCA variance across components instead of
# equalizing noisy trailing components with StandardScaler.
# Scale globally using the 99th percentile of absolute training
# PCA scores so dominant components drive larger quantum rotations.
pca_scale = float(np.percentile(np.abs(X_train_pca), 99))
if pca_scale <= 0.0:
    pca_scale = 1.0


def encode_angles(pca_features, scale):
    # Smoothly map variance-preserved PCA scores into [-pi, pi].
    return np.pi * np.tanh(pca_features / (1.5 * scale))


X_train = encode_angles(X_train_pca, pca_scale)
X_val = encode_angles(X_val_pca, pca_scale)
X_test = encode_angles(X_test_pca, pca_scale)


# ============================================================
# 5. BUILD THE 8-QUBIT HIERARCHICAL QCNN CIRCUIT (UPDATED)
# ============================================================

# Architecture:
# - Input Encoding: 16 PCA features on 8 qubits (RY: 0..7, RZ: 8..15)
# - Stage 1: Convolution on 8 qubits -> Pooling 8 -> 4 qubits
# - Stage 2: Convolution on 4 qubits -> Pooling 4 -> 2 qubits
# - Stage 3: Convolution on 2 qubits -> Pooling 2 -> 1 qubit (qubit 0)
# - Final single-qubit rotation on surviving qubit 0

x_params = ParameterVector("x", N_PCA)

# Parameter count calculation:
# Stage 1 (8 active): Conv (8*2=16) + Pool (4*2=8) = 24
# Stage 2 (4 active): Conv (4*2=8)  + Pool (2*2=4) = 12
# Stage 3 (2 active): Conv (2*2=4)  + Pool (1*2=2) = 6
# Final readout rotation on qubit 0: 2 (RY, RZ)
# Total trainable weights = 44
N_WEIGHTS = 44
theta = ParameterVector("theta", N_WEIGHTS)

qc = QuantumCircuit(N_QUBITS)
parameter_index = 0

# --- 5A. Quantum Data Encoding (All 16 PCA Features on 8 Qubits) ---
for qubit in range(N_QUBITS):
    # First 8 primary PCA components encoded via RY
    qc.ry(x_params[qubit], qubit)
    # Next 8 secondary PCA components encoded via RZ
    qc.rz(x_params[N_QUBITS + qubit], qubit)


def apply_conv_layer(circuit, active_qubits, params, idx):
    """
    2-qubit local convolutional block across even and odd neighbor pairs.
    """
    n_active = len(active_qubits)

    # Pre-entanglement trainable rotations
    for q in active_qubits:
        circuit.ry(params[idx], q)
        idx += 1
        circuit.rz(params[idx], q)
        idx += 1

    # Even pairs entanglement
    for i in range(0, n_active - 1, 2):
        circuit.cx(active_qubits[i], active_qubits[i + 1])

    # Odd pairs entanglement (circular boundary if > 2 qubits)
    if n_active > 2:
        for i in range(1, n_active - 1, 2):
            circuit.cx(active_qubits[i], active_qubits[i + 1])
        circuit.cx(active_qubits[-1], active_qubits[0])

    return idx


def apply_pool_layer(circuit, source_qubits, target_qubits, params, idx):
    """
    Quantum pooling block: transfers information from source_qubits
    into target_qubits via controlled rotations, halving active wires.
    """
    for src, tgt in zip(source_qubits, target_qubits):
        circuit.crz(params[idx], src, tgt)
        idx += 1
        circuit.cry(params[idx], src, tgt)
        idx += 1

    return idx


# --- Stage 1: 8 Qubits -> 4 Qubits ---
active_stage1 = [0, 1, 2, 3, 4, 5, 6, 7]
parameter_index = apply_conv_layer(
    qc, active_stage1, theta, parameter_index
)
# Pool wires [1, 3, 5, 7] into surviving wires [0, 2, 4, 6]
parameter_index = apply_pool_layer(
    qc, [1, 3, 5, 7], [0, 2, 4, 6], theta, parameter_index
)

# --- Stage 2: 4 Qubits -> 2 Qubits ---
active_stage2 = [0, 2, 4, 6]
parameter_index = apply_conv_layer(
    qc, active_stage2, theta, parameter_index
)
# Pool wires [2, 6] into surviving wires [0, 4]
parameter_index = apply_pool_layer(
    qc, [2, 6], [0, 4], theta, parameter_index
)

# --- Stage 3: 2 Qubits -> 1 Qubit ---
active_stage3 = [0, 4]
parameter_index = apply_conv_layer(
    qc, active_stage3, theta, parameter_index
)
# Pool wire [4] into final surviving wire [0]
parameter_index = apply_pool_layer(
    qc, [4], [0], theta, parameter_index
)

# --- Final Readout Adjustment on Surviving Qubit 0 ---
qc.ry(theta[parameter_index], 0)
parameter_index += 1
qc.rz(theta[parameter_index], 0)
parameter_index += 1

assert parameter_index == N_WEIGHTS

print("\nQuantum CNN model (8 -> 4 -> 2 -> 1)")
print("Qubits:", N_QUBITS)
print("PCA features:", N_PCA)
print("Hierarchical stages:", N_LAYERS)
print("Trainable parameters:", N_WEIGHTS)
print(qc.draw(output="text"))


# ============================================================
# 6. SINGLE-QUBIT POOLED QUANTUM READOUT (UPDATED)
# ============================================================

# Measure probability of |1> on the final pooled wire (qubit 0).
# In Qiskit's little-endian basis ordering, qubit 0 is the least
# significant bit (index & 1 == 1).

readout_weights = np.array(
    [
        float(index & 1)
        for index in range(2 ** N_QUBITS)
    ],
    dtype=np.float64,
)


def predict_probability_digit2(features, weights):
    probabilities_out = np.empty(len(features), dtype=np.float64)

    weight_bindings = {
        theta[i]: float(weights[i])
        for i in range(N_WEIGHTS)
    }

    for row, sample in enumerate(features):
        bindings = {
            x_params[i]: float(sample[i])
            for i in range(N_PCA)
        }
        bindings.update(weight_bindings)

        bound_circuit = qc.assign_parameters(
            bindings, inplace=False
        )

        state = Statevector.from_instruction(bound_circuit)
        probabilities = state.probabilities()

        # Probability of measuring |1> on pooled qubit 0 (class 1 = digit 2).
        p_digit2 = float(probabilities @ readout_weights)

        probabilities_out[row] = np.clip(
            p_digit2, 1e-7, 1.0 - 1e-7
        )

    return probabilities_out


# ============================================================
# 7. LOSS AND EVALUATION (WITH HINGE METRICS)
# ============================================================

def binary_cross_entropy(labels, probabilities):
    probabilities = np.clip(
        probabilities, 1e-7, 1.0 - 1e-7
    )

    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1 - labels) * np.log(1 - probabilities)
        )
    )


def compute_hinge_metrics(labels, probabilities, threshold=0.5):
    """
    Computes Hinge Accuracy and Hinge Loss overall and for each digit.
    - Maps binary labels {0, 1} to bipolar labels {-1, +1}.
    - Maps probabilities [0, 1] centered around `threshold` to [-1, +1] scores.
    """
    y_hinge = np.where(labels == 1, 1.0, -1.0)
    y_score = 2.0 * (probabilities - threshold)

    margins = y_hinge * y_score
    hinge_losses = np.maximum(0.0, 1.0 - margins)
    hinge_correct = (margins > 0.0).astype(np.float64)

    mask_digit1 = (labels == 0)
    mask_digit2 = (labels == 1)

    hinge_acc_overall = float(np.mean(hinge_correct))
    hinge_acc_digit1 = (
        float(np.mean(hinge_correct[mask_digit1]))
        if np.any(mask_digit1) else 0.0
    )
    hinge_acc_digit2 = (
        float(np.mean(hinge_correct[mask_digit2]))
        if np.any(mask_digit2) else 0.0
    )

    hinge_loss_overall = float(np.mean(hinge_losses))
    hinge_loss_digit1 = (
        float(np.mean(hinge_losses[mask_digit1]))
        if np.any(mask_digit1) else 0.0
    )
    hinge_loss_digit2 = (
        float(np.mean(hinge_losses[mask_digit2]))
        if np.any(mask_digit2) else 0.0
    )

    return {
        "hinge_accuracy": hinge_acc_overall,
        "hinge_accuracy_digit1": hinge_acc_digit1,
        "hinge_accuracy_digit2": hinge_acc_digit2,
        "hinge_loss": hinge_loss_overall,
        "hinge_loss_digit1": hinge_loss_digit1,
        "hinge_loss_digit2": hinge_loss_digit2,
    }


def evaluate(features, labels, weights, threshold=0.5):
    probabilities = predict_probability_digit2(features, weights)

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    loss = binary_cross_entropy(labels, probabilities)
    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(
        labels, predictions
    )
    f1 = f1_score(labels, predictions, zero_division=0)
    hinge_metrics = compute_hinge_metrics(
        labels, probabilities, threshold=threshold
    )

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
# 8. BALANCED MINI-BATCH SAMPLING
# ============================================================

train_class0 = np.flatnonzero(y_train == 0)
train_class1 = np.flatnonzero(y_train == 1)


def make_balanced_batch(generator):
    samples_per_class = BATCH_SIZE // 2

    idx0 = generator.choice(
        train_class0,
        size=samples_per_class,
        replace=False,
    )

    idx1 = generator.choice(
        train_class1,
        size=samples_per_class,
        replace=False,
    )

    indices = np.concatenate([idx0, idx1])
    generator.shuffle(indices)

    return X_train[indices], y_train[indices]


# ============================================================
# 9. SPSA GRADIENT ESTIMATION
# ============================================================

def estimate_spsa_gradient(
    features,
    labels,
    weights,
    generator,
    step,
):
    # Gradually decrease the perturbation.
    c_k = SPSA_C / (step ** 0.101)

    gradient = np.zeros_like(weights)

    for _ in range(SPSA_DIRECTIONS):
        delta = generator.choice(
            [-1.0, 1.0], size=len(weights)
        )

        loss_plus = binary_cross_entropy(
            labels,
            predict_probability_digit2(
                features, weights + c_k * delta
            ),
        )

        loss_minus = binary_cross_entropy(
            labels,
            predict_probability_digit2(
                features, weights - c_k * delta
            ),
        )

        directional_derivative = (
            (loss_plus - loss_minus) / (2.0 * c_k)
        )

        gradient += directional_derivative * delta

    gradient /= SPSA_DIRECTIONS

    # Limit exceptionally large noisy gradient estimates.
    gradient_norm = np.linalg.norm(gradient)

    if gradient_norm > 10.0:
        gradient *= 10.0 / gradient_norm

    return gradient


# ============================================================
# 10. TRAINING WITH SPSA + ADAM-STYLE UPDATES
# ============================================================

best_global_val_loss = float("inf")
best_global_weights = None
best_global_val_accuracy = 0.0
best_global_restart = None

history = []
start_time = time.time()

for restart in range(1, RESTARTS + 1):

    print("\n" + "=" * 60)
    print(f"TRAINING RESTART {restart}/{RESTARTS}")
    print("=" * 60)

    restart_rng = np.random.default_rng(
        SEED + 100 * restart
    )

    # Different initial values help avoid a poor initialization.
    weights = restart_rng.uniform(
        -0.5, 0.5, size=N_WEIGHTS
    )

    first_moment = np.zeros(N_WEIGHTS)
    second_moment = np.zeros(N_WEIGHTS)

    optimizer_step = 0
    learning_rate = ADAM_LR

    best_restart_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        for _ in range(STEPS_PER_EPOCH):
            optimizer_step += 1

            batch_X, batch_y = make_balanced_batch(
                restart_rng
            )

            gradient = estimate_spsa_gradient(
                batch_X,
                batch_y,
                weights,
                restart_rng,
                optimizer_step,
            )

            # Adam-style first and second moments
            first_moment = (
                ADAM_BETA1 * first_moment
                + (1.0 - ADAM_BETA1) * gradient
            )

            second_moment = (
                ADAM_BETA2 * second_moment
                + (1.0 - ADAM_BETA2) * gradient**2
            )

            m_hat = first_moment / (
                1.0 - ADAM_BETA1**optimizer_step
            )

            v_hat = second_moment / (
                1.0 - ADAM_BETA2**optimizer_step
            )

            weights -= learning_rate * (
                m_hat / (np.sqrt(v_hat) + ADAM_EPS)
            )

            # Rotation gates are periodic, so wrap rather than
            # clipping all parameters against a hard boundary.
            weights = (
                (weights + np.pi) % (2.0 * np.pi)
            ) - np.pi

        train_metrics = evaluate(
            X_train, y_train, weights
        )

        val_metrics = evaluate(
            X_val, y_val, weights
        )

        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]

        history.append([
            restart,
            epoch,
            train_loss,
            train_metrics["accuracy"],
            val_loss,
            val_metrics["accuracy"],
            learning_rate,
        ])

        print(
            f"Restart {restart} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"Val Hinge Acc (D1/D2): {val_metrics['hinge_accuracy_digit1']:.4f}/{val_metrics['hinge_accuracy_digit2']:.4f} | "
            f"LR: {learning_rate:.5f}"
        )

        # Select models by validation loss, not test accuracy.
        if val_loss < best_restart_val_loss - 1e-4:
            best_restart_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if val_loss < best_global_val_loss:
            best_global_val_loss = val_loss
            best_global_weights = weights.copy()
            best_global_val_accuracy = val_metrics["accuracy"]
            best_global_restart = restart

            np.save(
                OUTPUT_DIR / "best_quantum_weights.npy",
                best_global_weights,
            )

        # Reduce the learning rate when progress stalls.
        if patience_counter in (3, 5):
            learning_rate = max(
                learning_rate * 0.5, MIN_LR
            )

        if patience_counter >= PATIENCE:
            print(
                "Early stopping for this restart: "
                "validation loss stopped improving."
            )
            break


# ============================================================
# 11. SAVE PREPROCESSING AND TRAINING HISTORY
# ============================================================

if best_global_weights is None:
    raise RuntimeError(
        "Training did not produce a valid model checkpoint."
    )

joblib.dump(
    {
        "pca": pca,
        "pca_scale": pca_scale,
        "n_qubits": N_QUBITS,
        "n_pca": N_PCA,
        "n_layers": N_LAYERS,
    },
    OUTPUT_DIR / "preprocessing.joblib",
)

np.savetxt(
    OUTPUT_DIR / "training_history.csv",
    np.asarray(history, dtype=np.float64),
    delimiter=",",
    header=(
        "restart,epoch,train_loss,train_accuracy,"
        "val_loss,val_accuracy,learning_rate"
    ),
    comments="",
)


# ============================================================
# 12. OPTIONAL DECISION-THRESHOLD SELECTION
# ============================================================

# Choose a threshold using the validation set only.
# It may improve classification accuracy, but does not change
# the underlying quantum probabilities or BCE loss.

val_probabilities = predict_probability_digit2(
    X_val, best_global_weights
)

threshold_candidates = np.arange(0.30, 0.701, 0.01)

threshold_results = []

for threshold in threshold_candidates:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(np.int64)

    val_accuracy = accuracy_score(
        y_val, val_predictions
    )

    threshold_results.append(
        (val_accuracy, -abs(threshold - 0.5), threshold)
    )

_, _, best_threshold = max(threshold_results)

print("\nSelected threshold:", round(float(best_threshold), 2))


# ============================================================
# 13. FINAL TEST EVALUATION
# Run only after selecting the model and threshold.
# ============================================================

test_probabilities = predict_probability_digit2(
    X_test, best_global_weights
)

test_predictions = (
    test_probabilities >= best_threshold
).astype(np.int64)

test_loss = binary_cross_entropy(
    y_test, test_probabilities
)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test, test_predictions
)

test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

# Compute Hinge Metrics at both standard 0.5 and selected threshold
test_hinge_standard = compute_hinge_metrics(
    y_test, test_probabilities, threshold=0.5
)
test_hinge_selected = compute_hinge_metrics(
    y_test, test_probabilities, threshold=best_threshold
)

print("\n" + "=" * 60)
print("FINAL 8-QUBIT QUANTUM CNN (QCNN) RESULTS")
print("=" * 60)

print("Best restart:", best_global_restart)
print("Qubits:", N_QUBITS)
print("PCA features:", N_PCA)
print("Trainable quantum parameters:", N_WEIGHTS)

print(
    "Best validation loss:",
    round(float(best_global_val_loss), 4),
)
print(
    "Validation accuracy at best loss:",
    round(float(best_global_val_accuracy), 4),
)
print("Decision threshold:", round(float(best_threshold), 2))
print("Test loss:", round(float(test_loss), 4))
print("Test accuracy:", round(float(test_accuracy), 4))
print(
    "Test balanced accuracy:",
    round(float(test_balanced_accuracy), 4),
)
print("Test F1:", round(float(test_f1), 4))

print("\n--- HINGE METRICS (Standard Threshold = 0.50) ---")
print(
    "Overall Hinge Accuracy:",
    round(test_hinge_standard["hinge_accuracy"], 4),
)
print(
    "Digit 1 Hinge Accuracy:",
    round(test_hinge_standard["hinge_accuracy_digit1"], 4),
)
print(
    "Digit 2 Hinge Accuracy:",
    round(test_hinge_standard["hinge_accuracy_digit2"], 4),
)
print(
    "Overall Hinge Loss:",
    round(test_hinge_standard["hinge_loss"], 4),
)
print(
    "Digit 1 Hinge Loss:",
    round(test_hinge_standard["hinge_loss_digit1"], 4),
)
print(
    "Digit 2 Hinge Loss:",
    round(test_hinge_standard["hinge_loss_digit2"], 4),
)

print(
    f"\n--- HINGE METRICS (Selected Threshold = {best_threshold:.2f}) ---"
)
print(
    "Overall Hinge Accuracy:",
    round(test_hinge_selected["hinge_accuracy"], 4),
)
print(
    "Digit 1 Hinge Accuracy:",
    round(test_hinge_selected["hinge_accuracy_digit1"], 4),
)
print(
    "Digit 2 Hinge Accuracy:",
    round(test_hinge_selected["hinge_accuracy_digit2"], 4),
)
print(
    "Overall Hinge Loss:",
    round(test_hinge_selected["hinge_loss"], 4),
)
print(
    "Digit 1 Hinge Loss:",
    round(test_hinge_selected["hinge_loss_digit1"], 4),
)
print(
    "Digit 2 Hinge Loss:",
    round(test_hinge_selected["hinge_loss_digit2"], 4),
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Digit 1", "Digit 2"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )
)

np.savez(
    OUTPUT_DIR / "test_results.npz",
    y_true=y_test,
    y_pred=test_predictions,
    y_probability=test_probabilities,
    threshold=best_threshold,
    test_loss=test_loss,
    test_accuracy=test_accuracy,
    hinge_accuracy=test_hinge_selected["hinge_accuracy"],
    hinge_accuracy_digit1=test_hinge_selected["hinge_accuracy_digit1"],
    hinge_accuracy_digit2=test_hinge_selected["hinge_accuracy_digit2"],
    hinge_loss=test_hinge_selected["hinge_loss"],
    hinge_loss_digit1=test_hinge_selected["hinge_loss_digit1"],
    hinge_loss_digit2=test_hinge_selected["hinge_loss_digit2"],
    best_weights=best_global_weights,
)

print(
    "\nElapsed minutes:",
    round((time.time() - start_time) / 60, 2),
)
print("Saved results to:", OUTPUT_DIR.resolve())