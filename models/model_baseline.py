#model_baseline.py
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random

# ─── Đường dẫn ─────────────────────────────────────────────────────
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()

SCRIPT_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data/real_vs_fake"
OUT_DIR = SCRIPT_DIR / "model_baseline_artifacts"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE, BATCH, EPOCHS = 128, 64, 10

# ─── Transform ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ─── DataLoader ────────────────────────────────────────────────────
def get_dataloaders():
    def get_subset(dataset):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        subset_size = int(len(indices) * 0.3)
        return Subset(dataset, indices[:subset_size])

    train_dataset = get_subset(datasets.ImageFolder(DATA_DIR / "train", transform=transform))
    val_dataset = get_subset(datasets.ImageFolder(DATA_DIR / "valid", transform=transform))
    test_dataset = get_subset(datasets.ImageFolder(DATA_DIR / "test", transform=transform))

    return (
        DataLoader(train_dataset, batch_size=BATCH, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH),
        DataLoader(test_dataset, batch_size=BATCH)
    )

# ─── Mô hình mạng nơ-ron ──────────────────────────────────────────
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE * IMG_SIZE * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ─── Train và Eval ─────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer):
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1).float()
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((out > 0.5) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        history["accuracy"].append(acc)
        history["loss"].append(total_loss / len(train_loader))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_correct, val_total = 0, 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1).float()
                out = model(x)
                val_loss += criterion(out, y).item()
                val_correct += ((out > 0.5) == y).sum().item()
                val_total += y.size(0)

            val_acc = val_correct / val_total
            history["val_accuracy"].append(val_acc)
            history["val_loss"].append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{EPOCHS} | acc={acc:.3f} | val_acc={val_acc:.3f}")
    return history

# ─── Plot và lưu kết quả ──────────────────────────────────────────
def plot_and_save_history(hist):
    json.dump(hist, open(OUT_DIR / "hist.json", "w"))

    plt.figure(figsize=(5, 3))
    plt.plot(hist["accuracy"], label="train")
    plt.plot(hist["val_accuracy"], label="val")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "acc_curve.png"); plt.close()

    plt.figure(figsize=(5, 3))
    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png"); plt.close()

# ─── Hàm main ──────────────────────────────────────────────────────
def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    model = SimpleNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = train_model(model, train_loader, val_loader, criterion, optimizer)
    plot_and_save_history(history)

    # Evaluate
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss, test_correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1).float()
            out = model(x)
            test_loss += criterion(out, y).item()
            test_correct += ((out > 0.5) == y).sum().item()
            total += y.size(0)

    print(f"\n  Test accuracy = {test_correct / total:.3f} | loss = {test_loss / len(test_loader):.3f}")
    print("  All outputs →", OUT_DIR.resolve())

if __name__ == "__main__":
    main()

#  Test accuracy = 0.751 | loss = 0.505