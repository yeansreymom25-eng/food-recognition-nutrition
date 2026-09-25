import os
import torch
import torch.nn as nn
import torch.optim as optim

from data import create_dataloaders
from model import create_mobilenetv2


# --------------------------------------------------
# Project settings
# --------------------------------------------------

DATA_PATH = "split_data"
MODEL_DIR = "model"

NUM_CLASSES = 20
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

(
    train_loader,
    validation_loader,
    test_loader,
    class_names
) = create_dataloaders(
    DATA_PATH,
    batch_size=BATCH_SIZE
)

print("Number of classes:", len(class_names))


# --------------------------------------------------
# Load MobileNetV2
# --------------------------------------------------

model = create_mobilenetv2(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True
)

model = model.to(device)

print("MobileNetV2 ready!")


# --------------------------------------------------
# Loss function and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

print("Loss function: CrossEntropyLoss")
print("Optimizer: Adam")
print("Learning rate:", LEARNING_RATE)


# --------------------------------------------------
# Create model folder
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Training history
# --------------------------------------------------

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

best_val_accuracy = 0.0


# --------------------------------------------------
# Train for 5 epochs
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):

    # ==============================================
    # TRAINING
    # ==============================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )


    # ==============================================
    # VALIDATION
    # ==============================================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = (
        val_running_loss /
        len(validation_loader)
    )

    val_accuracy = (
        100 *
        val_correct /
        val_total
    )


    # ==============================================
    # SAVE RESULTS
    # ==============================================

    train_losses.append(
        train_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_losses.append(
        val_loss
    )

    val_accuracies.append(
        val_accuracy
    )


    # ==============================================
    # PRINT EPOCH RESULT
    # ==============================================

    print(
        f"\nEpoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f} | "
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # ==============================================
    # SAVE BEST MODEL
    # ==============================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_DIR,
                "best_mobilenetv2.pth"
            )
        )

        print(
            "New best model saved!"
        )

    print("-" * 50)


# --------------------------------------------------
# Finish
# --------------------------------------------------

print("\nTraining finished!")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)
