import copy
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from tqdm import tqdm
from model import LeafDiseaseCNN
from dataset import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeafDiseaseCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
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
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader.dataset), correct / total

train_losses = []
val_losses = []

best_val = float("inf")
patience = 3
wait = 0
best_weights =None


for epoch in range(1, 11):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch}:" f" Train Loss: {train_loss:.4f}" f" Val Loss: {val_loss:.4f}" f" Val Acc: {val_acc*100:.2f}%")
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

torch.save(model.state_dict(), "models/leaf_cnn_best.pth")

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_plot.png")
plt.show()