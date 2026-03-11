from pathlib import Path
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

train_dir = BASE_DIR / "data" / "fer2013" / "train"
test_dir = BASE_DIR / "data" / "fer2013" / "test"
model_save_path = BASE_DIR / "models" / "best_image_model_resnet18.pth"

selected_classes = ["angry", "fear", "happy", "sad", "surprise"]
class_names = selected_classes

batch_size = 32
epochs = 10
learning_rate = 1e-3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Kullanılan cihaz:", device)
print("Train path:", train_dir)
print("Test path:", test_dir)


# -------------------------------------------------
# TRANSFORMS
# ResNet pretrained weights -> 224x224, RGB, normalize
# -------------------------------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# -------------------------------------------------
# DATASETS
# -------------------------------------------------
train_dataset_full = datasets.ImageFolder(
    root=str(train_dir),
    transform=train_transform
)

test_dataset_full = datasets.ImageFolder(
    root=str(test_dir),
    transform=eval_transform
)

print("Tüm sınıflar:", train_dataset_full.classes)

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

train_size = int(0.8 * len(train_dataset_filtered))
val_size = len(train_dataset_filtered) - train_size

train_dataset, val_dataset = random_split(
    train_dataset_filtered,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# validation split için augmentation kapat
val_dataset.dataset.dataset.transform = eval_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Train split:", len(train_dataset))
print("Validation split:", len(val_dataset))


# -------------------------------------------------
# LABEL MAP
# -------------------------------------------------
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


# -------------------------------------------------
# MODEL: RESNET18 TRANSFER LEARNING
# -------------------------------------------------
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)



# son katmanı 5 sınıfa uyarla
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -------------------------------------------------
# TRAIN
# -------------------------------------------------
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
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
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), model_save_path)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"- Loss: {running_loss:.4f} "
        f"- Train Accuracy: {train_acc:.4f} "
        f"- Val Accuracy: {val_acc:.4f}"
    )

print(f"En iyi validation accuracy: {best_val_acc:.4f}")


# -------------------------------------------------
# TEST
# -------------------------------------------------
model.load_state_dict(best_model_wts)
model.eval()

all_labels = []
all_preds = []
correct = 0
total = 0

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

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("ResNet18 Transfer Learning - Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"Model kaydedildi: {model_save_path}")