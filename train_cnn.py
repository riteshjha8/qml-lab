
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# --------------------------------------------------
# 1. Configuration and reproducibility
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "downscaled_dataset" / "downscaled_data.npz"
QUANTUM_PATH = ROOT / "quantum_data" / "quantum_features.npz"
OUTPUT_DIR = ROOT / "cnn_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --------------------------------------------------
# 2. Load images and original train/test split
# --------------------------------------------------
data = np.load(DATA_PATH)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

quantum_data = np.load(QUANTUM_PATH)

original_train_indices = quantum_data[
    "train_original_indices"
].copy()

test_indices = quantum_data[
    "test_original_indices"
].copy()

blank_indices = quantum_data[
    "excluded_blank_indices"
].copy()

# Check the dataset dimensions.
assert X.ndim == 3 and X.shape[1:] == (16, 16)
assert len(X) == len(y)

# Verify that the saved indices cover all retained samples.
valid_indices = np.setdiff1d(
    np.arange(len(y)),
    blank_indices,
)

assert len(original_train_indices) + len(test_indices) == len(
    valid_indices
)

assert np.intersect1d(
    original_train_indices, test_indices
).size == 0

assert np.array_equal(
    np.sort(np.concatenate([
        original_train_indices,
        test_indices,
    ])),
    valid_indices,
)

# Split the training partition into training and validation.
train_indices, val_indices = train_test_split(
    original_train_indices,
    test_size=0.15,
    random_state=SEED,
    stratify=y[original_train_indices],
)

# --------------------------------------------------
# 3. Prepare image tensors
# --------------------------------------------------
X_train = X[train_indices, None, :, :]
y_train = y[train_indices]

X_val = X[val_indices, None, :, :]
y_val = y[val_indices]

X_test = X[test_indices, None, :, :]
y_test = y[test_indices]

# Check for overlap between all partitions.
assert np.intersect1d(train_indices, val_indices).size == 0
assert np.intersect1d(train_indices, test_indices).size == 0
assert np.intersect1d(val_indices, test_indices).size == 0

print("\nDATASET SUMMARY")
print("----------------")
print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)
print("Testing images:", X_test.shape)
print("Training classes:", np.unique(y_train))
print("Validation classes:", np.unique(y_val))
print("Testing classes:", np.unique(y_test))

# --------------------------------------------------
# 4. Create data loaders
# --------------------------------------------------
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
)

generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# --------------------------------------------------
# 5. Define the classical CNN
# --------------------------------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = DigitCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# --------------------------------------------------
# 6. Training and validation
# --------------------------------------------------
history = []

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_state = None

start_time = time.perf_counter()

for epoch in range(EPOCHS):

    # ----- Training -----
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # ----- Validation -----
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy:.4f}"
    )

    # Save the model with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        torch.save(
            best_state,
            OUTPUT_DIR / "cnn_model.pt",
        )

    else:
        epochs_without_improvement += 1

    # Stop if validation loss has not improved.
    if epochs_without_improvement >= PATIENCE:
        print(
            f"Early stopping: validation loss did not "
            f"improve for {PATIENCE} consecutive epochs."
        )
        break

training_seconds = time.perf_counter() - start_time

assert best_state is not None

# Restore the best validation checkpoint.
model.load_state_dict(best_state)
model.eval()

# --------------------------------------------------
# 7. Save training history
# --------------------------------------------------
with open(
    OUTPUT_DIR / "training_history.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(history, f, indent=2)

# --------------------------------------------------
# 8. Final test evaluation
# --------------------------------------------------
test_loss_sum = 0.0
test_correct = 0
test_total = 0

all_predictions = []
all_labels = []

test_start_time = time.perf_counter()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)

        test_loss_sum += loss.item() * labels.size(0)
        test_correct += (
            predictions == labels
        ).sum().item()
        test_total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_labels.extend(
            labels.cpu().tolist()
        )

inference_seconds = time.perf_counter() - test_start_time

test_loss = test_loss_sum / test_total
test_accuracy = test_correct / test_total

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(10)),
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(10)),
)

metrics = {
    "model": "Classical CNN",
    "seed": SEED,
    "epochs_completed": len(history),
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "testing_samples": len(X_test),
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "macro_f1": f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    ),
    "training_time_seconds": training_seconds,
    "test_inference_time_seconds": inference_seconds,
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
}

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f, indent=2)

# --------------------------------------------------
# 9. Print final results
# --------------------------------------------------
print("\nCNN TEST RESULTS")
print("----------------")
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(
    f"Macro F1: {metrics['macro_f1']:.4f}"
)
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Test inference time: {inference_seconds:.2f} seconds")
print("\nClassification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        zero_division=0,
    )
)

print("\nSaved files:")
print(OUTPUT_DIR / "cnn_model.pt")
print(OUTPUT_DIR / "training_history.json")
print(OUTPUT_DIR / "test_metrics.json")
