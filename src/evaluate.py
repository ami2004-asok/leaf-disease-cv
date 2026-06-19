import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision import models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from dataset import val_loader, CLASS_NAMES

# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# Recreate ResNet18 Architecture
# -----------------------------
weights = models.ResNet18_Weights.IMAGENET1K_V1

model = models.resnet18(weights=weights)

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 4)

model = model.to(device)

# -----------------------------
# Load Trained Weights
# -----------------------------
model.load_state_dict(
    torch.load(
        "models/resnet18_best.pth",
        map_location=device
    )
)

model.eval()

# -----------------------------
# Storage
# -----------------------------
all_preds = []
all_labels = []

os.makedirs("reports/errors", exist_ok=True)

error_count = 0

# -----------------------------
# Evaluation Loop
# -----------------------------
with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        # Save misclassified samples
        for i in range(len(images)):

            if preds[i] != labels[i]:

                if error_count < 10:

                    img = images[i].cpu()

                    mean = torch.tensor(
                        [0.485, 0.456, 0.406]
                    ).view(3, 1, 1)

                    std = torch.tensor(
                        [0.229, 0.224, 0.225]
                    ).view(3, 1, 1)

                    img = img * std + mean

                    img = (
                        img.permute(1, 2, 0)
                        .numpy()
                    )

                    img = np.clip(img, 0, 1)

                    true_label = CLASS_NAMES[
                        labels[i].item()
                    ]

                    pred_label = CLASS_NAMES[
                        preds[i].item()
                    ]

                    plt.figure(figsize=(4, 4))

                    plt.imshow(img)

                    plt.title(
                        f"True: {true_label}\n"
                        f"Pred: {pred_label}"
                    )

                    plt.axis("off")

                    plt.savefig(
                        f"reports/errors/error_{error_count}.png",
                        bbox_inches="tight"
                    )

                    plt.close()

                    error_count += 1

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

plt.figure(figsize=(8, 6))

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix")

plt.savefig(
    "reports/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES
)

print("\nClassification Report:\n")
print(report)

with open(
    "reports/classification_report.txt",
    "w"
) as f:
    f.write(report)

print(
    "\nResults saved to reports/"
)