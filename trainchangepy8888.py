
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
OUTPUT_DIR = Path("pure_qnn_8qubit_final_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pure variational quantum neural network
N_QUBITS = 8

# Two blocks of eight PCA features are encoded across two layers.
N_PCA = 16
N_LAYERS = N_PCA // N_QUBITS

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

assert N_PCA == N_QUBITS * N_LAYERS


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
# 4. PCA AND FEATURE SCALING
# ============================================================

pca = PCA(n_components=N_PCA, random_state=SEED)

X_train_pca = pca.fit_transform(X_train_raw)
X_val_pca = pca.transform(X_val_raw)
X_test_pca = pca.transform(X_test_raw)

print(
    "\nPCA variance retained:",
    round(float(pca.explained_variance_ratio_.sum()), 4),
)

# Standardize using the training set only.
feature_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train_pca)
X_val_scaled = feature_scaler.transform(X_val_pca)
X_test_scaled = feature_scaler.transform(X_test_pca)


def encode_angles(features):
    # Smoothly map standardized PCA scores into [-pi, pi].
    return np.pi * np.tanh(features / 2.0)


X_train = encode_angles(X_train_scaled)
X_val = encode_angles(X_val_scaled)
X_test = encode_angles(X_test_scaled)


# ============================================================
# 5. BUILD THE PURE 8-QUBIT QUANTUM CIRCUIT
# ============================================================

# Each circuit layer receives a different block of eight PCA
# features. With 16 features and 8 qubits, there are two blocks.

x_params = ParameterVector("x", N_PCA)

N_WEIGHTS = N_LAYERS * N_QUBITS * 2
theta = ParameterVector("theta", N_WEIGHTS)

qc = QuantumCircuit(N_QUBITS)

parameter_index = 0

for layer in range(N_LAYERS):

    feature_offset = layer * N_QUBITS

    # Quantum data encoding
    for qubit in range(N_QUBITS):
        feature = x_params[feature_offset + qubit]

        qc.ry(feature, qubit)
        qc.rz(feature, qubit)

    # Trainable quantum rotations
    for qubit in range(N_QUBITS):
        qc.ry(theta[parameter_index], qubit)
        parameter_index += 1

        qc.rz(theta[parameter_index], qubit)
        parameter_index += 1

    # Ring entanglement
    if layer % 2 == 0:
        for qubit in range(N_QUBITS - 1):
            qc.cx(qubit, qubit + 1)

        qc.cx(N_QUBITS - 1, 0)
    else:
        qc.cx(0, N_QUBITS - 1)

        for qubit in range(N_QUBITS - 1, 0, -1):
            qc.cx(qubit, qubit - 1)


print("\nQuantum model")
print("Qubits:", N_QUBITS)
print("PCA features:", N_PCA)
print("Quantum layers:", N_LAYERS)
print("Trainable parameters:", N_WEIGHTS)
print(qc.draw(output="text"))


# ============================================================
# 6. QUANTUM READOUT
# ============================================================

# Average the probability of measuring |1> over all eight
# qubits. This uses all qubits and adds no trainable classical
# neural-network branch.

readout_weights = np.array(
    [
        index.bit_count() / N_QUBITS
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

        # Probability of class 1 (original digit 2).
        p_digit2 = float(probabilities @ readout_weights)

        probabilities_out[row] = np.clip(
            p_digit2, 1e-7, 1.0 - 1e-7
        )

    return probabilities_out


# ============================================================
# 7. LOSS AND EVALUATION
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

    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "predictions": predictions,
        "probabilities": probabilities,
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
        "feature_scaler": feature_scaler,
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

print("\n" + "=" * 60)
print("FINAL PURE 8-QUBIT QNN RESULTS")
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
    best_weights=best_global_weights,
)

print(
    "\nElapsed minutes:",
    round((time.time() - start_time) / 60, 2),
)
print("Saved results to:", OUTPUT_DIR.resolve())
