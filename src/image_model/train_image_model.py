from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Cihaz
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# Proje kökü
BASE_DIR = Path(__file__).resolve().parents[2]

train_dir = BASE_DIR / "data" / "fer2013" / "train"
test_dir = BASE_DIR / "data" / "fer2013" / "test"

print("Train path:", train_dir)
print("Test path:", test_dir)

# Eğitim için augmentation
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Test/validation için sade transform
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Tüm dataset
train_dataset_full = datasets.ImageFolder(
    root=str(train_dir),
    transform=train_transform
)

test_dataset_full = datasets.ImageFolder(
    root=str(test_dir),
    transform=test_transform
)

print("Tüm sınıflar:", train_dataset_full.classes)

selected_classes = ["angry", "fear", "happy", "sad", "surprise"]

selected_train_indices = [
    i for i, (_, label) in enumerate(train_dataset_full.samples)
    if train_dataset_full.classes[label] in selected_classes
]

selected_test_indices = [
    i for i, (_, label) in enumerate(test_dataset_full.samples)
    if test_dataset_full.classes[label] in selected_classes
]

train_dataset_filtered = Subset(train_dataset_full, selected_train_indices)
test_dataset = Subset(test_dataset_full, selected_test_indices)

print("Filtrelenmiş train image sayısı:", len(train_dataset_filtered))
print("Filtrelenmiş test image sayısı:", len(test_dataset))
print("Kullanılan sınıflar:", selected_classes)

# Train / validation split
train_size = int(0.8 * len(train_dataset_filtered))
val_size = len(train_dataset_filtered) - train_size

train_dataset, val_dataset = random_split(
    train_dataset_filtered,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Validation setinde augmentation istemiyoruz
# random_split sonrası Subset içindeki kaynak dataset transformunu değiştiriyoruz
val_dataset.dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Train split:", len(train_dataset))
print("Validation split:", len(val_dataset))

# Label mapping
selected_class_to_new_idx = {
    "angry": 0,
    "fear": 1,
    "happy": 2,
    "sad": 3,
    "surprise": 4
}

def remap_labels(batch_labels, dataset_full):
    remapped = []
    for label in batch_labels:
        old_class_name = dataset_full.classes[label]
        remapped.append(selected_class_to_new_idx[old_class_name])
    return torch.tensor(remapped, dtype=torch.long)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = EmotionCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
best_val_acc = 0.0

for epoch in range(epochs):
    # TRAIN
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        labels = remap_labels(labels.tolist(), train_dataset_full)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            labels = remap_labels(labels.tolist(), train_dataset_full)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BASE_DIR / "models" / "best_image_model.pth")

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"- Loss: {running_loss:.4f} "
        f"- Train Accuracy: {train_acc:.4f} "
        f"- Val Accuracy: {val_acc:.4f}"
    )

print(f"En iyi validation accuracy: {best_val_acc:.4f}")

# En iyi modeli yükle
model.load_state_dict(torch.load(BASE_DIR / "models" / "best_image_model.pth", map_location=device))
model.eval()

# TEST
# En iyi modeli yükle
model.load_state_dict(torch.load(BASE_DIR / "models" / "best_image_model.pth", map_location=device))
model.eval()

# TEST
correct = 0
total = 0

all_labels = []
all_preds = []

idx_to_class = {v: k for k, v in selected_class_to_new_idx.items()}

with torch.no_grad():
    for images, labels in test_loader:
        labels = remap_labels(labels.tolist(), test_dataset_full)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")

class_names = ["angry", "fear", "happy", "sad", "surprise"]

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Image Model Confusion Matrix")
plt.colorbar()

tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()