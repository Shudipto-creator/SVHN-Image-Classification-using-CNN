
"""
Topic 2: SVHN Image Classification using CNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ------------------------------
# Hyperparameters
# ------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
DATA_ROOT = "./data"
OUTPUT_DIR = Path("./outputs_topic2")
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

# ------------------------------
# CNN Architecture
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
# Training and evaluation functions
# ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

# ------------------------------
# Main execution guard
# ------------------------------
if __name__ == '__main__':
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load data (automatically downloads if not present)
    print("Loading SVHN dataset...")
    train_dataset = torchvision.datasets.SVHN(
        root=DATA_ROOT, split='train', download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.SVHN(
        root=DATA_ROOT, split='test', download=True, transform=test_transform
    )

    # Use num_workers=0 to avoid multiprocessing issues on Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model, loss, optimizer, scheduler
    model = SVHN_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Training loop
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0

    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step(test_acc)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_svhn_cnn.pth")
            print(f"  -> New best model saved (acc={test_acc:.4f})")

    # Save final model
    torch.save(model.state_dict(), MODEL_DIR / "final_svhn_cnn.pth")
    print(f"\nTraining completed. Best test accuracy: {best_acc:.4f}")

    # Plot curves
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves.png", dpi=150)
    plt.show()
    print(f"Training curves saved to {PLOT_DIR / 'training_curves.png'}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(MODEL_DIR / "best_svhn_cnn.pth", map_location=device))
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test accuracy (best model): {final_acc:.4f} ({final_acc*100:.2f}%)")

    # Sample predictions
    model.eval()
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:10].to(device)
    sample_labels = sample_labels[:10].cpu().numpy()
    with torch.no_grad():
        outputs = model(sample_images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    for i in range(10):
        img = sample_images[i].cpu().numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5  # denormalize
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"True: {sample_labels[i]}, Pred: {preds[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sample_predictions.png", dpi=150)
    plt.show()
    print(f"Sample predictions saved to {PLOT_DIR / 'sample_predictions.png'}")

    # Print architecture diagram
    print("\n" + "="*60)
    print("CNN Architecture Diagram (text description):")
    print("="*60)
    print("""
Input: 32x32x3 (RGB)
    |
Conv2d(3->32, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 16x16x32
    |
Conv2d(32->64, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 8x8x64
    |
Conv2d(64->128, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 4x4x128
    |
Flatten → 2048
    |
Linear(2048->256) → BatchNorm → ReLU → Dropout(0.5)
    |
Linear(256->128) → BatchNorm → ReLU → Dropout(0.5)
    |
Linear(128->10) → logits (0-9)
    """)

    print("\nProject completed successfully.")
    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")