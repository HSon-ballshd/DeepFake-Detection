#model_evolution.py
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
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
OUT_DIR = SCRIPT_DIR / "model_evolution_artifacts"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE, BATCH, EPOCHS = 128, 64, 10

# ─── Transform ─────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ─── DataLoader ────────────────────────────────────────────────────
def get_dataloaders():
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "valid", transform=val_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=val_transform)

    return (
        DataLoader(train_dataset, batch_size=BATCH, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH),
        DataLoader(test_dataset, batch_size=BATCH)
    )

# ─── Mô hình ConvNeXt ──────────────────────────────────────────────
class ConvNeXtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.convnext_tiny(pretrained=True)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

# ─── Train và Eval ─────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_acc = 0
    

    for epoch in range(EPOCHS):
        # Training
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

        scheduler.step()

        # Save best model & early stopping
        patience = 3
        no_improve = 0
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "best_convnext_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        print(f"Epoch {epoch+1}/{EPOCHS} | acc={acc:.3f} | val_acc={val_acc:.3f}")
    
    print(f"Best validation accuracy: {best_acc:.3f}")
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
    model = ConvNeXtModel()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
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