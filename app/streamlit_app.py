from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.utils.text_preprocessing import clean_text


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
TEXT_MODEL_PATH = BASE_DIR / "models" / "text_model.pkl"
TFIDF_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
IMAGE_MODEL_PATH = BASE_DIR / "models" / "best_image_model.pth"

CLASS_NAMES = ["angry", "fear", "happy", "sad", "surprise"]

TEXT_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -------------------------------------------------
# IMAGE MODEL
# -------------------------------------------------
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


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    text_model = joblib.load(TEXT_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)

    image_model = EmotionCNN(num_classes=5)
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    image_model.to(device)
    image_model.eval()

    return text_model, vectorizer, image_model


text_model, vectorizer, image_model = load_models()


# -------------------------------------------------
# TRANSFORM
# -------------------------------------------------
image_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def label_to_turkish(label: str) -> str:
    mapping = {
        "angry": "Kızgın",
        "fear": "Korku",
        "happy": "Mutlu",
        "sad": "Üzgün",
        "surprise": "Sürpriz",
    }
    return mapping.get(label, label)


def probs_to_dict(probs: np.ndarray) -> dict:
    return {
        label_to_turkish(label): float(prob)
        for label, prob in zip(CLASS_NAMES, probs)
    }


# -------------------------------------------------
# PREDICTION FUNCTIONS
# -------------------------------------------------
def predict_text(text: str) -> np.ndarray:
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    probs = text_model.predict_proba(X)[0]
    return probs


def predict_image(pil_image: Image.Image) -> np.ndarray:
    image = image_transform(pil_image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(image)
        probs = torch.softmax(outputs, dim=1)

    return probs.cpu().numpy()[0]


# -------------------------------------------------
# CONFIDENCE-AWARE FUSION
# -------------------------------------------------
def fuse_predictions(text_probs: np.ndarray, image_probs: np.ndarray):
    text_idx = int(np.argmax(text_probs))
    image_idx = int(np.argmax(image_probs))

    text_conf = float(text_probs[text_idx])
    image_conf = float(image_probs[image_idx])

    text_label = CLASS_NAMES[text_idx]
    image_label = CLASS_NAMES[image_idx]

    # 1) Aynı tahminse direkt kabul et
    if text_label == image_label:
        final_probs = (text_probs + image_probs) / 2
        final_label = text_label
        reason = "Metin ve görsel model aynı duygu tahminini verdi."
        return final_probs, final_label, reason, text_conf, image_conf

    # 2) Text modeli çok güvenliyse text'i seç
    if text_conf >= 0.60:
        final_probs = text_probs
        final_label = text_label
        reason = "Metin modeli daha yüksek güvenle tahmin yaptı."
        return final_probs, final_label, reason, text_conf, image_conf

    # 3) Görsel model çok güvenliyse ve text kararsızsa görseli seç
    if image_conf >= 0.75 and text_conf < 0.60:
        final_probs = image_probs
        final_label = image_label
        reason = "Görsel model daha yüksek güvenle tahmin yaptı."
        return final_probs, final_label, reason, text_conf, image_conf

    # 4) İkisi de kararsızsa weighted average
    final_probs = TEXT_WEIGHT * text_probs + IMAGE_WEIGHT * image_probs
    final_idx = int(np.argmax(final_probs))
    final_label = CLASS_NAMES[final_idx]
    reason = "İki modelin olasılıkları ağırlıklı olarak birleştirildi."
    return final_probs, final_label, reason, text_conf, image_conf


# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="Çok Kipli Duygu Tanıma", layout="wide")

st.title("Çok Kipli Duygu Tanıma İstemi")
st.write("Metin + yüz görseli kullanarak duygu tahmini yapar.")

col1, col2 = st.columns(2)

with col1:
    user_text = st.text_area(
        "Metni gir",
        placeholder="Örn: Bugün kendimi çok kötü hissediyorum..."
    )

with col2:
    uploaded_image = st.file_uploader(
        "Yüz görseli yükle",
        type=["jpg", "jpeg", "png"]
    )

analyze_clicked = st.button("Analiz Et")

if analyze_clicked:
    if not user_text.strip():
        st.error("Lütfen bir metin gir.")
        st.stop()

    if uploaded_image is None:
        st.error("Lütfen bir görsel yükle.")
        st.stop()

    try:
        pil_image = Image.open(uploaded_image).convert("RGB")
    except Exception as e:
        st.error(f"Görsel açılamadı: {e}")
        st.stop()

    text_probs = predict_text(user_text)
    image_probs = predict_image(pil_image)

    text_pred = CLASS_NAMES[int(np.argmax(text_probs))]
    image_pred = CLASS_NAMES[int(np.argmax(image_probs))]

    final_probs, fusion_pred, fusion_reason, text_conf, image_conf = fuse_predictions(
        text_probs, image_probs
    )

    st.subheader("Sonuçlar")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown("### Metin Modeli")
        st.write(f"**Tahmin:** {label_to_turkish(text_pred)}")
        st.write(f"**Güven:** {text_conf:.2f}")
        st.json(probs_to_dict(text_probs))

    with result_col2:
        st.markdown("### Görsel Model")
        st.write(f"**Tahmin:** {label_to_turkish(image_pred)}")
        st.write(f"**Güven:** {image_conf:.2f}")
        st.image(pil_image, caption="Yüklenen görsel", width=220)
        st.json(probs_to_dict(image_probs))

    with result_col3:
        st.markdown("### Fusion Sonucu")
        st.write(f"**Nihai Tahmin:** {label_to_turkish(fusion_pred)}")
        st.write(f"**Karar Nedeni:** {fusion_reason}")
        st.json(probs_to_dict(final_probs))

    st.success(f"Nihai duygu tahmini: {label_to_turkish(fusion_pred)}")