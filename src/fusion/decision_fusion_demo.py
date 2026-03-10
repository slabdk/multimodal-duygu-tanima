from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import joblib
import numpy as np

# -----------------------------------
# CONFIG
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

TEXT_MODEL_PATH = BASE_DIR / "models" / "text_model.pkl"
TFIDF_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
IMAGE_MODEL_PATH = BASE_DIR / "models" / "best_image_model.pth"

class_names = ["angry", "fear", "happy", "sad", "surprise"]

# fusion ağırlıkları
TEXT_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------------
# IMAGE MODEL
# -----------------------------------

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

# -----------------------------------
# MODELLERİ YÜKLE
# -----------------------------------

print("Modeller yükleniyor...")

text_model = joblib.load(TEXT_MODEL_PATH)
vectorizer = joblib.load(TFIDF_PATH)

image_model = EmotionCNN(num_classes=5)
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
image_model.to(device)
image_model.eval()

print("Modeller hazır.")

# -----------------------------------
# IMAGE TRANSFORM
# -----------------------------------

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# -----------------------------------
# TEXT PREDICTION
# -----------------------------------

def predict_text(text):

    X = vectorizer.transform([text])
    probs = text_model.predict_proba(X)[0]

    return probs

# -----------------------------------
# IMAGE PREDICTION
# -----------------------------------

def predict_image(image_path):

    image = Image.open(image_path)

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(image)
        probs = torch.softmax(outputs, dim=1)

    return probs.cpu().numpy()[0]

# -----------------------------------
# FUSION
# -----------------------------------

def fusion_prediction(text, image_path):

    text_probs = predict_text(text)
    image_probs = predict_image(image_path)

    final_probs = TEXT_WEIGHT * text_probs + IMAGE_WEIGHT * image_probs

    pred_idx = np.argmax(final_probs)
    pred_label = class_names[pred_idx]

    return {
        "text_probs": text_probs,
        "image_probs": image_probs,
        "final_probs": final_probs,
        "prediction": pred_label
    }

# -----------------------------------
# DEMO
# -----------------------------------

if __name__ == "__main__":

    sample_text = "bugün çok mutluyum hayat harika"

    sample_image = BASE_DIR / "data" / "fer2013" / "test" / "happy" / "PrivateTest_10077120.jpg"

    result = fusion_prediction(sample_text, sample_image)

    print("\nTEXT PROBS")
    print(dict(zip(class_names, result["text_probs"])))

    print("\nIMAGE PROBS")
    print(dict(zip(class_names, result["image_probs"])))

    print("\nFUSION PROBS")
    print(dict(zip(class_names, result["final_probs"])))

    print("\nFINAL EMOTION:", result["prediction"])