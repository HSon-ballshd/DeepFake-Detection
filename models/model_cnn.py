# model_cnn_pytorch.py
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

# ── paths ─────────────────────────────────────────────────────────
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data/real_vs_fake"  # Giống như model_evolution.py
OUT_DIR = SCRIPT_DIR / "model_cnn_artifacts"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE, BATCH, EPOCHS = 128, 64, 10          # nhỏ hơn để debug nhanh
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── transforms ────────────────────────────────────────────────────
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

# ── dataset ────────────────────────────────────────────────────────
def get_dataloaders(subset_frac=0.3):
    """Tạo DataLoader giống như model_evolution.py"""
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "valid", transform=val_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=val_transform)
    
    # Subset for debug (giống như subset_frac trong code cũ)
    if subset_frac < 1.0:
        train_size = int(len(train_dataset) * subset_frac)
        val_size = int(len(val_dataset) * subset_frac)
        
        train_indices = random.sample(range(len(train_dataset)), train_size)
        val_indices = random.sample(range(len(val_dataset)), val_size)
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    return (
        DataLoader(train_dataset, batch_size=BATCH, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH, shuffle=False),
        DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    )

# ── cnn model ─────────────────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size after convolutions
        # IMG_SIZE=128 -> 128/2/2/2 = 16, so feature maps are 16x16x128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def export_architecture(model):
    # Save model architecture as text
    arch_info = {
        "model_structure": str(model),
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(OUT_DIR / "cnn_arch.json", "w") as f:
        json.dump(arch_info, f, indent=2)
    
    print(f"Model has {arch_info['total_params']:,} total parameters")
    print(f"Model has {arch_info['trainable_params']:,} trainable parameters")

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

def train_model(model, train_loader, val_loader, criterion, optimizer):
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    model.to(DEVICE)
    best_acc = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1).float()
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
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1).float()
                out = model(x)
                val_loss += criterion(out, y).item()
                val_correct += ((out > 0.5) == y).sum().item()
                val_total += y.size(0)

            val_acc = val_correct / val_total
            history["val_accuracy"].append(val_acc)
            history["val_loss"].append(val_loss / len(val_loader))

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "best_cnn_model.pth")
            
        print(f"Epoch {epoch+1}/{EPOCHS} | acc={acc:.3f} | val_acc={val_acc:.3f}")
    
    print(f"Best validation accuracy: {best_acc:.3f}")
    return history

# ── main ───────────────────────────────────────────────────────────
def main():
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(subset_frac=0.3)
    
    # Initialize model
    model = CNNModel().to(DEVICE)
    export_architecture(model)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    torch.save(model, OUT_DIR / "cnn_model_complete.pth")  # Save complete model
    
    with open(OUT_DIR / "hist.json", "w") as f:
        json.dump(history, f, indent=2)
    
    plot_and_save_history(history)
    
    # Test evaluation (use validation logic)
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        test_loss, test_correct, test_total = 0, 0, 0
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1).float()
            out = model(x)
            test_loss += criterion(out, y).item()
            test_correct += ((out > 0.5) == y).sum().item()
            test_total += y.size(0)
        test_acc = test_correct / test_total
        test_loss = test_loss / len(test_loader)
    print(f"\n  Test accuracy = {test_acc:.3f} | loss = {test_loss:.3f}")
    print("  outputs →", OUT_DIR.resolve())

if __name__ == "__main__":
    main()

#  Test accuracy = 0.751 | loss = 0.505