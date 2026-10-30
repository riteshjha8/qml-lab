
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "downscaled_dataset_28" / "downscaled_data_28.npz"
)

QUANTUM_PATH = (
    ROOT / "quantum_data_28" / "quantum_features_28.npz"
)

OUTPUT_DIR = ROOT / "cnn_results_28"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 0.001
NUM_CLASSES = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# 2. Load data and saved splits
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not QUANTUM_PATH.exists():
    raise FileNotFoundError(
        f"Quantum split file not found: {QUANTUM_PATH}"
    )

data = np.load(DATA_PATH)
quantum_data = np.load(QUANTUM_PATH)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

train_indices = quantum_data[
    "train_original_indices"
].astype(np.int64)

val_indices = quantum_data[
    "val_original_indices"
].astype(np.int64)

test_indices = quantum_data[
    "test_original_indices"
].astype(np.int64)

excluded_indices = quantum_data[
    "excluded_blank_indices"
].astype(np.int64)

print(f"Dataset shape: {X.shape}")
print(f"Training:      {len(train_indices)}")
print(f"Validation:    {len(val_indices)}")
print(f"Testing:       {len(test_indices)}")
print(f"Excluded:      {len(excluded_indices)}")


# ============================================================
# 3. Validate splits
# ============================================================

if X.shape[1:] != (28, 28):
    raise ValueError(f"Expected 28x28 images, got {X.shape}")

if len(X) != len(y):
    raise ValueError("Image and label counts differ.")

for name, indices in [
    ("training", train_indices),
    ("validation", val_indices),
    ("testing", test_indices),
    ("excluded", excluded_indices),
]:
    if np.any(indices < 0) or np.any(indices >= len(y)):
        raise ValueError(f"Invalid {name} indices.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate {name} indices.")

if np.intersect1d(train_indices, val_indices).size:
    raise ValueError("Training and validation overlap.")

if np.intersect1d(train_indices, test_indices).size:
    raise ValueError("Training and testing overlap.")

if np.intersect1d(val_indices, test_indices).size:
    raise ValueError("Validation and testing overlap.")

if any(
    np.intersect1d(indices, excluded_indices).size
    for indices in [train_indices, val_indices, test_indices]
):
    raise ValueError("Excluded samples appear in a data split.")

expected_indices = np.setdiff1d(
    np.arange(len(y)),
    excluded_indices,
)

actual_indices = np.sort(
    np.concatenate([train_indices, val_indices, test_indices])
)

if not np.array_equal(actual_indices, expected_indices):
    raise ValueError(
        "CNN split does not match the quantum preprocessing split."
    )

# Labels and ordering must match the quantum dataset's source.
original_data = np.load(
    ROOT / "downscaled_dataset" / "downscaled_data.npz"
)
if not np.array_equal(y, original_data["y"]):
    raise ValueError("Label order differs from the original dataset.")

del original_data

# Ensure all ten classes are represented.
if not np.array_equal(np.unique(y), np.arange(NUM_CLASSES)):
    raise ValueError("Expected digit labels 0 through 9.")

print("Split validation passed.")


# ============================================================
# 4. Prepare PyTorch tensors
# ============================================================

# CNN input format: (samples, channels, height, width)
X = X[:, None, :, :]

X_train = torch.from_numpy(X[train_indices])
y_train = torch.from_numpy(y[train_indices])

X_val = torch.from_numpy(X[val_indices])
y_val = torch.from_numpy(y[val_indices])

X_test = torch.from_numpy(X[test_indices])
y_test = torch.from_numpy(y[test_indices])

# Use a seeded generator for reproducible shuffling.
generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
    num_workers=0,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 5. Define a stronger CNN for 28x28 images
# ============================================================

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            # Block 3: 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = DigitCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 6. Training and evaluation helpers
# ============================================================

def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


# ============================================================
# 7. Train with early stopping
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
history = []

model_path = OUTPUT_DIR / "cnn_model_28.pt"

print("\nStarting CNN training...")
start_training = time.perf_counter()

for epoch in range(1, MAX_EPOCHS + 1):
    epoch_start = time.perf_counter()

    train_loss, train_accuracy = run_epoch(
        train_loader, training=True
    )

    val_loss, val_accuracy = run_epoch(
        val_loader, training=False
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_seconds = time.perf_counter() - epoch_start

    record = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "learning_rate": float(current_lr),
        "epoch_seconds": float(epoch_seconds),
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy * 100:.2f}% | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"{epoch_seconds:.1f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": val_accuracy,
                "seed": SEED,
            },
            model_path,
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping: validation loss did not improve "
            f"for {PATIENCE} epochs."
        )
        break

training_seconds = time.perf_counter() - start_training

with open(
    OUTPUT_DIR / "training_history_28.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(history, file, indent=2)


# ============================================================
# 8. Load best validation checkpoint
# ============================================================

checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")


# ============================================================
# 9. Evaluate the test set once
# ============================================================

all_predictions = []
all_targets = []

# Warm up the model before measuring inference speed.
with torch.no_grad():
    warmup = X_test[:min(32, len(X_test))].to(DEVICE)
    _ = model(warmup)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

inference_start = time.perf_counter()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions.tolist())
        all_targets.extend(labels.numpy().tolist())

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

inference_seconds = time.perf_counter() - inference_start

test_accuracy = accuracy_score(
    all_targets, all_predictions
)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0,
)

report = classification_report(
    all_targets,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
)

print("\n" + "=" * 60)
print("28x28 CNN TEST RESULTS")
print("=" * 60)
print(f"Test accuracy:      {test_accuracy * 100:.2f}%")
print(f"Macro F1-score:     {macro_f1 * 100:.2f}%")
print(f"Best epoch:         {best_epoch}")
print(f"Training time:      {training_seconds:.2f} seconds")
print(f"Test inference:     {inference_seconds:.4f} seconds")
print(
    f"Avg inference/img:  "
    f"{inference_seconds / len(test_indices) * 1000:.4f} ms"
)

print("\nClassification report:")
print(
    classification_report(
        all_targets,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(matrix)


# ============================================================
# 10. Save final results
# ============================================================

metrics = {
    "model": "DigitCNN_28x28",
    "input_shape": [28, 28],
    "train_samples": int(len(train_indices)),
    "validation_samples": int(len(val_indices)),
    "test_samples": int(len(test_indices)),
    "excluded_samples": int(len(excluded_indices)),
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_val_loss),
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(macro_f1),
    "training_seconds": float(training_seconds),
    "test_inference_seconds": float(inference_seconds),
    "average_inference_ms_per_image": float(
        inference_seconds / len(test_indices) * 1000
    ),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "device": str(DEVICE),
    "seed": SEED,
}

with open(
    OUTPUT_DIR / "test_metrics_28.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=2)

np.save(
    OUTPUT_DIR / "test_predictions_28.npy",
    np.asarray(all_predictions, dtype=np.int64),
)

np.save(
    OUTPUT_DIR / "test_targets_28.npy",
    np.asarray(all_targets, dtype=np.int64),
)

print("\nSaved output files:")
print(f"Model:       {model_path}")
print(f"History:     {OUTPUT_DIR / 'training_history_28.json'}")
print(f"Metrics:     {OUTPUT_DIR / 'test_metrics_28.json'}")
print(f"Predictions: {OUTPUT_DIR / 'test_predictions_28.npy'}")
print(f"Targets:     {OUTPUT_DIR / 'test_targets_28.npy'}")

print("\nCNN training and evaluation completed.")
