
from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


# ---------------- CONFIGURATION ----------------

DATA_PATH = Path("downscaled_dataset_28/downscaled_data_28.npz")
OUTPUT_DIR = Path("pure_qnn_1_vs_2_output")

N_QUBITS = 4
N_PCA = 4
N_LAYERS = 2

EPOCHS = 30
LEARNING_RATE = 0.003
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- LOAD DATA ----------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]

mask = np.isin(y, [1, 2])
X = X[mask].astype(np.float64)
y = y[mask]

# Map digit 1 -> class 0 and digit 2 -> class 1
y = (y == 2).astype(np.int64)

X = X.reshape(len(X), -1) / 255.0

print("Filtered dataset:", X.shape)
print("Digit 1:", np.sum(y == 0))
print("Digit 2:", np.sum(y == 1))


# ---------------- SPLIT DATA ----------------
# 50% train, 20% validation, 30% test.
# Stratification preserves class proportions.

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.6,
    random_state=SEED, stratify=y_temp
)


# ---------------- PCA PREPROCESSING ----------------

pca = PCA(n_components=N_PCA, random_state=SEED)

X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

# Convert PCA features to rotation angles in [-pi, pi].
# Fit scaling on training data only.

scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)) # type: ignore

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("PCA variance retained:", round(
    np.sum(pca.explained_variance_ratio_), 4
))


# ---------------- PURE QUANTUM CIRCUIT ----------------

# Data-encoding parameters
x_params = ParameterVector("x", N_QUBITS)

# Trainable variational parameters
theta = ParameterVector(
    "theta", N_LAYERS * N_QUBITS * 2
)

qc = QuantumCircuit(N_QUBITS)

# Encode four PCA features
for q in range(N_QUBITS):
    qc.ry(x_params[q], q)
    qc.h(q)

# Trainable quantum layers
k = 0
for layer in range(N_LAYERS):
    for q in range(N_QUBITS):
        qc.ry(theta[k], q)
        k += 1
        qc.rz(theta[k], q)
        k += 1

    # Entangle neighboring qubits in a ring
    for q in range(N_QUBITS - 1):
        qc.cx(q, q + 1)
    qc.cx(N_QUBITS - 1, 0)

# Measure the expectation value of Z on qubit 0.
# Statevector simulation computes this exactly.
print("\nQuantum circuit:")
print(qc.draw())


# ---------------- QUANTUM FORWARD PASS ----------------

def predict_probability_one(features, weights):
    """
    Return P(class 1), where class 1 represents digit 2.
    This is calculated from a quantum measurement probability.
    """
    probabilities = []

    for sample in features:
        bindings = {
            x_params[i]: float(sample[i])
            for i in range(N_QUBITS)
        }
        bindings.update({
            theta[i]: float(weights[i])
            for i in range(len(weights))
        })

        bound = qc.assign_parameters(bindings)
        state = Statevector.from_instruction(bound)

        # Qiskit statevector basis index bit 0 is qubit 0.
        # Probability that qubit 0 is measured as 1.
        p_one = 0.0
        for index, probability in enumerate(state.probabilities()):
            if (index & 1) == 1:
                p_one += probability

        probabilities.append(p_one)

    return np.asarray(probabilities)


def binary_cross_entropy(y_true, probabilities):
    eps = 1e-7
    p = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
    )


def evaluate(features, labels, weights):
    probs = predict_probability_one(features, weights)
    preds = (probs >= 0.5).astype(int)
    return (
        binary_cross_entropy(labels, probs),
        accuracy_score(labels, preds),
        f1_score(labels, preds, zero_division=0),
    )


# ---------------- TRAIN QUANTUM PARAMETERS ----------------

rng = np.random.default_rng(SEED)

n_weights = len(theta)
weights = rng.uniform(-0.1, 0.1, size=n_weights)

best_val_acc = -1.0
best_weights = weights.copy()

# Parameter-shift gradient for each Ry/Rz parameter.
# For rotations, derivative = [f(theta+pi/2)-f(theta-pi/2)]/2.
# The probability derivative is used in the BCE chain rule.

def loss_for_weights(w, features, labels):
    p = predict_probability_one(features, w)
    return binary_cross_entropy(labels, p)


def parameter_shift_gradient(features, labels, w):
    probs = predict_probability_one(features, w)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    # d(BCE)/d(p)
    dloss_dp = (probs - labels) / (probs * (1 - probs))
    dloss_dp /= len(labels)

    grad = np.zeros_like(w)
    shift = np.pi / 2

    for j in range(len(w)):
        plus = w.copy()
        minus = w.copy()
        plus[j] += shift
        minus[j] -= shift

        p_plus = predict_probability_one(features, plus)
        p_minus = predict_probability_one(features, minus)

        dp = (p_plus - p_minus) / 2.0
        grad[j] = np.sum(dloss_dp * dp)

    return grad


start = time.time()

for epoch in range(1, EPOCHS + 1):
    grad = parameter_shift_gradient(X_train, y_train, weights)
    weights -= LEARNING_RATE * grad

    train_loss, train_acc, _ = evaluate(
        X_train, y_train, weights
    )
    val_loss, val_acc, val_f1 = evaluate(
        X_val, y_val, weights
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train loss {train_loss:.4f} | "
        f"Train acc {train_acc:.4f} | "
        f"Val loss {val_loss:.4f} | "
        f"Val acc {val_acc:.4f} | "
        f"Val F1 {val_f1:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = weights.copy()
        np.save(OUTPUT_DIR / "best_quantum_weights.npy", best_weights)


# ---------------- FINAL TEST ----------------

weights = best_weights

test_loss, test_acc, test_f1 = evaluate(
    X_test, y_test, weights
)

test_probs = predict_probability_one(X_test, weights)
test_preds = (test_probs >= 0.5).astype(int)

print("\nBest validation accuracy:", round(best_val_acc, 4))
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_acc, 4))
print("Test macro F1:", round(test_f1, 4)) # type: ignore

print("\nClassification report:")
print(classification_report(
    y_test, test_preds,
    labels=[0, 1],
    target_names=["Digit 1", "Digit 2"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_preds, labels=[0, 1]))

print("Elapsed minutes:", round((time.time() - start) / 60, 2))

np.savez(
    OUTPUT_DIR / "test_results.npz",
    y_true=y_test,
    y_pred=test_preds,
    y_probability=test_probs,
    best_weights=weights,
)
