import copy
import torch
from torch import nn, optim
from torchvision import models
from tqdm import tqdm

from dataset import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 4
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Train only FC layer initially
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader.dataset), correct / total


best_val = float("inf")
best_weights = None

patience = 3
wait = 0

# -------- Phase 1: Train FC Layer --------

for epoch in range(1, 6):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device,
    )

    print(
        f"Epoch {epoch}: "
        f"Train Loss={train_loss:.4f} "
        f"Val Loss={val_loss:.4f} "
        f"Val Acc={val_acc*100:.2f}%"
    )

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        best_weights = copy.deepcopy(model.state_dict())
    else:
        wait += 1

        if wait >= patience:
            print("Early stopping triggered")
            break


# -------- Phase 2: Unfreeze Layer4 --------

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam(
    [
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.layer4.parameters(), "lr": 1e-5},
    ]
)

wait = 0

for epoch in range(6, 11):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device,
    )

    print(
        f"Epoch {epoch}: "
        f"Train Loss={train_loss:.4f} "
        f"Val Loss={val_loss:.4f} "
        f"Val Acc={val_acc*100:.2f}%"
    )

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        best_weights = copy.deepcopy(model.state_dict())
    else:
        wait += 1

        if wait >= patience:
            print("Early stopping triggered")
            break


model.load_state_dict(best_weights)

torch.save(
    model.state_dict(),
    "models/resnet18_best.pth"
)

print("Best model saved to models/resnet18_best.pth")