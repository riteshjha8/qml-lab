from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_FILE = Path("quantum_data_28/quantum_features_28.npz")
OUTPUT_DIR = Path("qcnn_results_28_optimized")

N_QUBITS = 8
N_CLASSES = 10

# STEP 1: Use more training data
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 1000

# Evaluate on the complete, unchanged test set
FULL_TEST_EVALUATION = True
TEST_SAMPLES = 4311  # Used only if full evaluation is disabled

BATCH_SIZE = 8
EPOCHS = 6
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
PATIENCE = 3

DEVICE = "cpu"

# Stop early if validation loss does not improve
MIN_DELTA = 1e-5


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA HELPERS
# ============================================================

def balanced_subset(X, y, requested_size, seed):
    """Choose approximately equal numbers of examples per class."""

    if requested_size > len(y):
        raise ValueError(
            f"Requested {requested_size} samples, "
            f"but only {len(y)} are available."
        )

    if requested_size < N_CLASSES:
        raise ValueError(
            "Subset must contain at least one example per class."
        )

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    if len(classes) != N_CLASSES:
        raise ValueError(
            f"Expected {N_CLASSES} classes, found {len(classes)}."
        )

    per_class = requested_size // N_CLASSES
    remainder = requested_size % N_CLASSES

    selected = []

    for class_position, class_id in enumerate(classes):
        class_indices = np.flatnonzero(y == class_id)

        # Distribute any remainder across the first classes.
        count = per_class + (
            1 if class_position < remainder else 0
        )

        if len(class_indices) < count:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} "
                f"examples, but {count} are required."
            )

        chosen = rng.choice(
            class_indices,
            size=count,
            replace=False,
        )
        selected.extend(chosen.tolist())

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)

    return X[selected], y[selected], selected


def class_counts(y):
    return np.bincount(
        np.asarray(y, dtype=np.int64),
        minlength=N_CLASSES,
    ).tolist()


def validate_arrays(X, y, name):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    if len(X) != len(y):
        raise ValueError(
            f"{name}: features and labels have different lengths."
        )

    if X.ndim != 2 or X.shape[1] != N_QUBITS:
        raise ValueError(
            f"{name}: expected features shaped (N, {N_QUBITS}), "
            f"received {X.shape}."
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{name}: features contain NaN or infinity.")

    if not np.isin(y, np.arange(N_CLASSES)).all():
        raise ValueError(f"{name}: labels must be digits 0 through 9.")

    return X, y


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE.resolve()}\n"
            "Run prepare_quantum_data_28.py first."
        )

    with np.load(DATA_FILE) as data:
        required = [
            "train_angles",
            "val_angles",
            "test_angles",
            "y_train",
            "y_val",
            "y_test",
        ]

        missing = [key for key in required if key not in data]

        if missing:
            raise KeyError(
                f"Missing dataset keys: {missing}. "
                "Check prepare_quantum_data_28.py."
            )

        X_train_all = data["train_angles"].copy()
        X_val_all = data["val_angles"].copy()
        X_test = data["test_angles"].copy()

        y_train_all = data["y_train"].copy()
        y_val_all = data["y_val"].copy()
        y_test = data["y_test"].copy()

    X_train_all, y_train_all = validate_arrays(
        X_train_all, y_train_all, "Training data"
    )
    X_val_all, y_val_all = validate_arrays(
        X_val_all, y_val_all, "Validation data"
    )
    X_test, y_test = validate_arrays(
        X_test, y_test, "Test data"
    )

    X_train, y_train, train_indices = balanced_subset(
        X_train_all,
        y_train_all,
        TRAIN_SAMPLES,
        SEED,
    )

    X_val, y_val, val_indices = balanced_subset(
        X_val_all,
        y_val_all,
        VAL_SAMPLES,
        SEED + 1,
    )

    if FULL_TEST_EVALUATION:
        test_indices = np.arange(len(y_test))
    else:
        test_size = min(TEST_SAMPLES, len(y_test))
        _, _, test_indices = balanced_subset(
            X_test,
            y_test,
            test_size,
            SEED + 2,
        )
        test_indices.sort()

    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    print("\nDataset summary")
    print("-" * 60)
    print(f"All available training examples: {len(y_train_all)}")
    print(f"All available validation examples: {len(y_val_all)}")
    print(f"Complete test set: {len(data_test_labels := y_test)}")
    print(f"Selected training examples: {len(y_train)}")
    print(f"Selected validation examples: {len(y_val)}")
    print(f"Test examples evaluated: {len(y_test)}")
    print(f"Training class counts: {class_counts(y_train)}")
    print(f"Validation class counts: {class_counts(y_val)}")
    print(f"Test class counts: {class_counts(y_test)}")

    # Save the selected original positions for reproducibility.
    np.savez_compressed(
        OUTPUT_DIR / "selected_data_indices.npz",
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

def build_quantum_model():
    """
    Build an 8-qubit variational quantum feature extractor.

    Each qubit receives one input angle.
    Each qubit has trainable RY and RZ rotations.
    A ring of CNOT gates entangles the qubits.
    Z expectation values become 8 classical features.

    This is a hybrid VQC, not a quantum convolution/pooling QCNN.
    """

    inputs = ParameterVector("x", N_QUBITS)
    weights = ParameterVector("theta", 2 * N_QUBITS)

    circuit = QuantumCircuit(N_QUBITS)

    # Input encoding
    for qubit in range(N_QUBITS):
        circuit.ry(inputs[qubit], qubit)

    # Trainable quantum layer
    for qubit in range(N_QUBITS):
        circuit.ry(weights[qubit], qubit)
        circuit.rz(weights[N_QUBITS + qubit], qubit)

    # Ring entanglement
    for qubit in range(N_QUBITS - 1):
        circuit.cx(qubit, qubit + 1)

    circuit.cx(N_QUBITS - 1, 0)

    # One Pauli-Z observable per qubit
    observables = []

    for qubit in range(N_QUBITS):
        pauli = ["I"] * N_QUBITS
        pauli[N_QUBITS - 1 - qubit] = "Z"

        observables.append(
            SparsePauliOp.from_list([
                ("".join(pauli), 1.0)
            ])
        )

    estimator = StatevectorEstimator()

    qnn = EstimatorQNN(
        circuit=circuit,
        estimator=estimator,
        observables=observables,
        input_params=list(inputs),
        weight_params=list(weights),
        input_gradients=False,
    )

    # Deterministic initial quantum weights
    generator = torch.Generator().manual_seed(SEED)
    initial_weights = (
        0.1
        * torch.randn(
            qnn.num_weights,
            generator=generator,
        )
    )

    quantum_layer = TorchConnector(
        qnn,
        initial_weights=initial_weights,
    )

    return circuit, quantum_layer


# ============================================================
# HYBRID CLASSIFIER
# ============================================================

class HybridVQC(nn.Module):
    def __init__(self, quantum_layer):
        super().__init__()

        self.quantum_layer = quantum_layer
        self.classifier = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        quantum_features = self.quantum_layer(x)

        # TorchConnector normally returns (batch, features).
        if quantum_features.ndim == 1:
            quantum_features = quantum_features.unsqueeze(0)

        return self.classifier(quantum_features)


# ============================================================
# METRICS AND EVALUATION
# ============================================================

def calculate_metrics(y_true, y_pred, average="macro"):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average=average, # type: ignore
                zero_division=0,
            )
        ),
    }


def evaluate(model, X, y, criterion, batch_size=BATCH_SIZE):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            end = min(start + batch_size, len(y))

            xb = torch.as_tensor(
                X[start:end],
                dtype=torch.float32,
                device=DEVICE,
            )

            yb = torch.as_tensor(
                y[start:end],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * len(yb)

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_labels.extend(yb.cpu().numpy().tolist())
            all_logits.append(logits.cpu().numpy())

    y_pred = np.asarray(all_predictions, dtype=np.int64)
    y_true = np.asarray(all_labels, dtype=np.int64)
    logits_array = np.concatenate(all_logits, axis=0)

    metrics = calculate_metrics(y_true, y_pred)
    metrics["loss"] = float(total_loss / len(y_true))

    return metrics, y_true, y_pred, logits_array


def save_training_plots(history):
    epochs = [entry["epoch"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        epochs,
        [entry["train_loss"] for entry in history],
        marker="o",
        label="Training loss",
    )
    axes[0].plot(
        epochs,
        [entry["val_loss"] for entry in history],
        marker="o",
        label="Validation loss",
    )
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        [entry["train_accuracy"] for entry in history],
        marker="o",
        label="Training accuracy",
    )
    axes[1].plot(
        epochs,
        [entry["val_accuracy"] for entry in history],
        marker="o",
        label="Validation accuracy",
    )
    axes[1].set_title("Accuracy by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "training_curves.png",
        dpi=150,
    )
    plt.close(fig)


def save_diagnostics(y_true, y_pred, logits):
    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set_title("Hybrid VQC Confusion Matrix")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))

    threshold = cm.max() / 2 if cm.size else 0

    for row in range(N_CLASSES):
        for col in range(N_CLASSES):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row, col] > threshold
                    else "black"
                ),
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    # Prediction distribution
    predicted_counts = np.bincount(
        y_pred,
        minlength=N_CLASSES,
    )
    actual_counts = np.bincount(
        y_true,
        minlength=N_CLASSES,
    )

    positions = np.arange(N_CLASSES)
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        positions - width / 2,
        actual_counts,
        width,
        label="Actual",
    )
    ax.bar(
        positions + width / 2,
        predicted_counts,
        width,
        label="Predicted",
    )

    ax.set_title("Actual vs Predicted Class Counts")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Number of images")
    ax.set_xticks(positions)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "prediction_distribution.png",
        dpi=150,
    )
    plt.close(fig)

    # Logit diagnostics
    probabilities = torch.softmax(
        torch.as_tensor(logits),
        dim=1,
    ).numpy()

    mean_probabilities = probabilities.mean(axis=0)
    mean_max_probability = float(
        probabilities.max(axis=1).mean()
    )
    prediction_entropy = float(
        -np.mean(
            np.sum(
                probabilities
                * np.log(np.clip(probabilities, 1e-12, 1.0)),
                axis=1,
            )
        )
    )

    print("\nPrediction diagnostics")
    print("-" * 60)
    print("Actual class counts:     ", actual_counts.tolist())
    print("Predicted class counts:  ", predicted_counts.tolist())
    print(
        "Mean predicted probability by digit:",
        np.round(mean_probabilities, 4).tolist(),
    )
    print(
        f"Mean maximum class probability: "
        f"{mean_max_probability:.4f}"
    )
    print(
        f"Mean prediction entropy: "
        f"{prediction_entropy:.4f} "
        f"(maximum for 10 classes is about 2.3026)"
    )

    print("\nConfusion matrix")
    print("Rows = actual digits; columns = predicted digits")
    print(cm)

    print("\nClassification report")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(N_CLASSES)),
            digits=4,
            zero_division=0,
        )
    )

    return cm, predicted_counts


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("=" * 68)
    print("HYBRID VARIATIONAL QUANTUM CLASSIFIER")
    print("Larger dataset + diagnostics + experiment comparison")
    print("=" * 68)

    print(f"Device: {DEVICE}")
    print(f"Training subset target: {TRAIN_SAMPLES}")
    print(f"Validation subset target: {VAL_SAMPLES}")
    print(f"Epoch limit: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Full test evaluation: {FULL_TEST_EVALUATION}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    X_train, y_train, X_val, y_val, X_test, y_test = (
        load_dataset()
    )

    print("\nBuilding quantum circuit...")

    circuit, quantum_layer = build_quantum_model()

    circuit_file = OUTPUT_DIR / "hybrid_vqc_circuit.txt"
    circuit_file.write_text(
        str(circuit.draw(output="text")),
        encoding="utf-8",
    )

    print(f"Circuit depth: {circuit.depth()}")
    print(
        f"Trainable quantum parameters: "
        f"{quantum_layer.neural_network.num_weights}"
    )

    model = HybridVQC(quantum_layer).to(DEVICE)

    quantum_parameter_count = sum(
        p.numel()
        for p in model.quantum_layer.parameters()
        if p.requires_grad
    )

    classical_parameter_count = sum(
        p.numel()
        for p in model.classifier.parameters()
        if p.requires_grad
    )

    total_parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Quantum trainable parameters: {quantum_parameter_count}")
    print(f"Classical trainable parameters: {classical_parameter_count}")
    print(f"Total trainable parameters: {total_parameter_count}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # STEP 2A: GRADIENT DIAGNOSTIC
    # --------------------------------------------------------

    print("\nChecking gradients on one training batch...")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    diagnostic_size = min(BATCH_SIZE, len(y_train))

    diagnostic_X = torch.as_tensor(
        X_train[:diagnostic_size],
        dtype=torch.float32,
        device=DEVICE,
    )
    diagnostic_y = torch.as_tensor(
        y_train[:diagnostic_size],
        dtype=torch.long,
        device=DEVICE,
    )

    diagnostic_logits = model(diagnostic_X)
    diagnostic_loss = criterion(
        diagnostic_logits,
        diagnostic_y,
    )
    diagnostic_loss.backward()

    quantum_grad_sq = 0.0
    classical_grad_sq = 0.0
    missing_quantum_grads = 0
    missing_classical_grads = 0

    for parameter in model.quantum_layer.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                missing_quantum_grads += 1
            else:
                quantum_grad_sq += (
                    parameter.grad.detach().norm().item() ** 2
                )

    for parameter in model.classifier.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                missing_classical_grads += 1
            else:
                classical_grad_sq += (
                    parameter.grad.detach().norm().item() ** 2
                )

    quantum_grad_norm = quantum_grad_sq ** 0.5
    classical_grad_norm = classical_grad_sq ** 0.5

    print(f"Diagnostic loss: {diagnostic_loss.item():.6f}")
    print(f"Quantum gradient norm: {quantum_grad_norm:.8f}")
    print(f"Classical gradient norm: {classical_grad_norm:.8f}")
    print(f"Missing quantum gradients: {missing_quantum_grads}")
    print(f"Missing classical gradients: {missing_classical_grads}")

    with open(
        OUTPUT_DIR / "gradient_diagnostics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "diagnostic_loss": diagnostic_loss.item(),
                "quantum_gradient_norm": quantum_grad_norm,
                "classical_gradient_norm": classical_grad_norm,
                "missing_quantum_gradients": missing_quantum_grads,
                "missing_classical_gradients": missing_classical_grads,
            },
            file,
            indent=2,
        )

    # Remove diagnostic gradients before the real training.
    optimizer.zero_grad(set_to_none=True)

    if quantum_grad_norm == 0.0:
        print(
            "\nWARNING: Quantum gradient norm is zero. "
            "The quantum parameters may not be updating effectively."
        )

    if classical_grad_norm == 0.0:
        print(
            "\nWARNING: Classical gradient norm is zero. "
            "The classifier may not be updating effectively."
        )

    # --------------------------------------------------------
    # STEP 1 + STEP 3: TRAIN AND RECORD COMPARABLE METRICS
    # --------------------------------------------------------

    print("\nStarting training...")
    print(
        "CPU quantum simulation can be slow. "
        "The epoch limit is a maximum, not a target to force."
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        model.train()

        permutation = np.random.default_rng(
            SEED + epoch
        ).permutation(len(y_train))

        total_train_loss = 0.0
        train_predictions = []
        train_labels = []

        for start in range(0, len(y_train), BATCH_SIZE):
            batch_indices = permutation[
                start:start + BATCH_SIZE
            ]

            xb = torch.as_tensor(
                X_train[batch_indices],
                dtype=torch.float32,
                device=DEVICE,
            )
            yb = torch.as_tensor(
                y_train[batch_indices],
                dtype=torch.long,
                device=DEVICE,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(yb)

            train_predictions.extend(
                logits.argmax(dim=1).detach().cpu().numpy().tolist()
            )
            train_labels.extend(yb.cpu().numpy().tolist())

        train_loss = total_train_loss / len(y_train)

        train_metrics = calculate_metrics(
            train_labels,
            train_predictions,
        )

        val_metrics, _, _, _ = evaluate(
            model,
            X_val,
            y_val,
            criterion,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": float(epoch_seconds),
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['macro_f1']:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

        # Save the complete history after each epoch.
        with open(
            OUTPUT_DIR / "training_history.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history, file, indent=2)

        save_training_plots(history)

        if val_metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                best_state,
                OUTPUT_DIR / "best_hybrid_vqc.pt",
            )

            print("  Best checkpoint updated.")

        else:
            epochs_without_improvement += 1
            print(
                f"  No meaningful validation-loss improvement "
                f"({epochs_without_improvement}/{PATIENCE})."
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation loss did not improve.")
            break

    training_seconds = time.perf_counter() - training_start

    if best_state is None:
        raise RuntimeError(
            "No best model checkpoint was produced."
        )

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training time: {training_seconds:.2f} seconds")

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating selected checkpoint...")

    test_start = time.perf_counter()

    test_metrics, y_true, y_pred, logits = evaluate(
        model,
        X_test,
        y_test,
        criterion,
    )

    test_seconds = time.perf_counter() - test_start

    cm, predicted_counts = save_diagnostics(
        y_true,
        y_pred,
        logits,
    )

    results = {
        "experiment": "hybrid_vqc_1000_training_samples",
        "seed": SEED,
        "architecture": "8-qubit VQC with classical linear classifier",
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_true)),
        "training_class_counts": class_counts(y_train),
        "validation_class_counts": class_counts(y_val),
        "test_class_counts": class_counts(y_true),
        "prediction_counts": predicted_counts.tolist(),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "training_seconds": float(training_seconds),
        "test_evaluation_seconds": float(test_seconds),
        "quantum_gradient_norm": float(quantum_grad_norm),
        "classical_gradient_norm": float(classical_grad_norm),
        "confusion_matrix": cm.tolist(),
        "trainable_quantum_parameters": int(quantum_parameter_count),
        "trainable_classical_parameters": int(classical_parameter_count),
        "total_trainable_parameters": int(total_parameter_count),
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        logits=logits,
        predicted_counts=np.asarray(predicted_counts),
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("FINAL RESULTS")
    print("=" * 68)
    print(f"Best epoch: {best_epoch}")
    print(f"Test samples: {len(y_true)}")
    print(f"Test loss: {test_metrics['loss']:.6f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f} "
          f"({100 * test_metrics['accuracy']:.2f}%)")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Test evaluation time: {test_seconds:.2f} seconds")
    print(f"Prediction counts by digit: {predicted_counts.tolist()}")

    print("\nSaved outputs:")
    print(f"  Model: {OUTPUT_DIR / 'best_hybrid_vqc.pt'}")
    print(f"  Metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"  History: {OUTPUT_DIR / 'training_history.json'}")
    print(f"  Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(
        f"  Prediction distribution: "
        f"{OUTPUT_DIR / 'prediction_distribution.png'}"
    )
    print(f"  Training curves: {OUTPUT_DIR / 'training_curves.png'}")
    print(
        f"  Test predictions: {OUTPUT_DIR / 'test_predictions.npz'}"
    )
    print(
        f"  Gradient diagnostics: "
        f"{OUTPUT_DIR / 'gradient_diagnostics.json'}"
    )
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
