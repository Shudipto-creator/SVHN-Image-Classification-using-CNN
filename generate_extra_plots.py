import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from pathlib import Path

# ------------------------------
# Configuration (same as original)
# ------------------------------
BATCH_SIZE = 64
DATA_ROOT = "./data"
MODEL_PATH = "./outputs_topic2/models/best_svhn_cnn.pth"
OUTPUT_DIR = Path("./extra_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Model definition (must match original exactly)
# ------------------------------
class SVHN_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.relu_fc2(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x

# ------------------------------
# Load test data
# ------------------------------
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = torchvision.datasets.SVHN(
    root=DATA_ROOT, split='test', download=True, transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ------------------------------
# Load model
# ------------------------------
model = SVHN_CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ------------------------------
# Get all predictions
# ------------------------------
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# ------------------------------
# 1. Full Confusion Matrix (10x10)
# ------------------------------
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Confusion Matrix on SVHN Test Set')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix_full.png', dpi=150)
plt.close()

# ------------------------------
# 2. Per‑class Precision/Recall Bar Chart
# ------------------------------
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
x = np.arange(10)
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, precision, width, label='Precision', color='steelblue')
ax.bar(x + width/2, recall, width, label='Recall', color='darkorange')
ax.set_xlabel('Digit Class')
ax.set_ylabel('Score')
ax.set_title('Per‑Class Precision and Recall')
ax.set_xticks(x)
ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9])
ax.set_ylim(0, 1.05)
ax.legend()
for i, (p, r) in enumerate(zip(precision, recall)):
    ax.text(i - width/2, p + 0.02, f'{p:.2f}', ha='center', fontsize=9)
    ax.text(i + width/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'per_class_precision_recall.png', dpi=150)
plt.close()

# ------------------------------
# 3. Misclassified Examples Grid
# ------------------------------
mis_idx = np.where(all_labels != all_preds)[0]
num_mis = min(16, len(mis_idx))  # 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.ravel()
for i in range(num_mis):
    idx = mis_idx[i]
    img = test_dataset[idx][0].cpu().numpy().transpose(1,2,0)
    img = img * 0.5 + 0.5  # denormalize
    img = np.clip(img, 0, 1)
    axes[i].imshow(img)
    axes[i].set_title(f'T:{all_labels[idx]} P:{all_preds[idx]}', fontsize=8)
    axes[i].axis('off')
for j in range(num_mis, 16):
    axes[j].axis('off')
plt.suptitle('Misclassified Test Images (True vs Predicted)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'misclassifications.png', dpi=150)
plt.close()

print(f"Extra plots saved in {OUTPUT_DIR.resolve()}")
print(f"Files generated: confusion_matrix_full.png, per_class_precision_recall.png, misclassifications.png")