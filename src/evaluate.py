import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data import create_dataloaders
from model import create_mobilenetv2


# --------------------------------------------------
# Project settings
# --------------------------------------------------

DATA_PATH = "split_data"
MODEL_DIR = "model"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_improved_mobilenetv2.pth"
)

BATCH_SIZE = 32
NUM_CLASSES = 20


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Create results folder
# --------------------------------------------------

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Load test dataset
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

print("Test images:", len(test_loader.dataset))


# --------------------------------------------------
# Load final improved model
# --------------------------------------------------

model = create_mobilenetv2(
    num_classes=NUM_CLASSES,
    pretrained=False,
    freeze_backbone=False
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Final improved model loaded!")


# --------------------------------------------------
# Predict test dataset
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# --------------------------------------------------
# Final test accuracy
# --------------------------------------------------

correct = sum(
    prediction == label
    for prediction, label in zip(
        all_predictions,
        all_labels
    )
)

final_accuracy = (
    correct /
    len(all_labels) *
    100
)

print("\nFinal Test Images:", len(all_labels))
print("Correct Predictions:", correct)

print(
    f"Final Test Accuracy: "
    f"{final_accuracy:.2f}%"
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

report_path = os.path.join(
    RESULTS_DIR,
    "final_classification_report.csv"
)

report_df.to_csv(
    report_path
)

print(
    "\nFinal classification report saved!"
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

fig, ax = plt.subplots(
    figsize=(14, 14)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90
)

plt.title(
    "Final Improved MobileNetV2 Confusion Matrix"
)

plt.tight_layout()

confusion_path = os.path.join(
    RESULTS_DIR,
    "final_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Final confusion matrix saved!"
)


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

comparison_df = pd.DataFrame({
    "Model": [
        "Original MobileNetV2",
        "Fine-Tuned MobileNetV2",
        "Improved MobileNetV2"
    ],

    "Test Accuracy (%)": [
        92.28,
        95.18,
        95.50
    ]
})

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\nModel Comparison")
print("----------------------")
print(comparison_df)

print(
    "\nModel comparison saved!"
)
