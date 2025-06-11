import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# Thêm imports cho metrics
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Định nghĩa transform như lúc training
IMG_SIZE = 128
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Tạo test dataset
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = REPO_ROOT / "data/realvsfake"  # Đường dẫn đến thư mục dữ liệu
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=64)

from torch import nn
from torchvision import models

# Định nghĩa lại kiến trúc model
class ConvNeXtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.convnext_tiny(pretrained=True)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

# Khởi tạo model và load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtModel()
model.load_state_dict(torch.load("model_evolution_artifacts_final/best_convnext_model.pth", map_location=device))
model.eval()
model.to(device)

criterion = nn.BCELoss()
test_loss, test_correct, total = 0, 0, 0

# Lists để lưu kết quả cho metrics
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1).float()
        out = model(x)
        test_loss += criterion(out, y).item()
        test_correct += ((out > 0.5) == y).sum().item()
        total += y.size(0)
        
        # Thu thập predictions và labels để tính metrics
        all_preds.extend(out.cpu().numpy().flatten())
        all_labels.extend(y.cpu().numpy().flatten())

# Chuyển sang numpy array
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Tính và in accuracy
accuracy = test_correct / total
print(f"Test accuracy = {accuracy:.3f} | loss = {test_loss / len(test_loader):.3f}")

# Tính và in AUC-ROC
auc_roc = roc_auc_score(all_labels, all_preds)
print(f"AUC-ROC = {auc_roc:.3f}")

# Tính Confusion Matrix
binary_preds = (all_preds > 0.5).astype(int)
cm = confusion_matrix(all_labels, binary_preds)
print("\nConfusion Matrix:")
print(cm)

# Tính và in precision, recall, F1-score
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"\nPrecision = {precision:.3f}")
print(f"Recall = {recall:.3f}")
print(f"F1-score = {f1:.3f}")

# Vẽ ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(all_labels, all_preds)
plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')

# Vẽ Confusion Matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()