from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

from src.utils.text_preprocessing import clean_text


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BERT_TEXT_MODEL_PATH = BASE_DIR / "models" / "berturk_emotion_model"
IMAGE_MODEL_PATH = BASE_DIR / "models" / "best_image_model_resnet18.pth"

CLASS_NAMES = ["angry", "fear", "happy", "sad", "surprise"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -------------------------------------------------
# IMAGE MODEL
# -------------------------------------------------
def build_image_model(num_classes=5):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    if not BERT_TEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"BERT text modeli bulunamadı: {BERT_TEXT_MODEL_PATH}")

    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Image modeli bulunamadı: {IMAGE_MODEL_PATH}")

    text_tokenizer = AutoTokenizer.from_pretrained(BERT_TEXT_MODEL_PATH)
    text_model = AutoModelForSequenceClassification.from_pretrained(BERT_TEXT_MODEL_PATH)
    text_model.to(device)
    text_model.eval()

    label_encoder = joblib.load(BERT_TEXT_MODEL_PATH / "label_encoder.pkl")

    image_model = build_image_model(num_classes=5)
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    image_model.to(device)
    image_model.eval()

    return text_tokenizer, text_model, label_encoder, image_model


text_tokenizer, text_model, label_encoder, image_model = load_models()


# -------------------------------------------------
# IMAGE TRANSFORM
# -------------------------------------------------
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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
        label_to_turkish(label): round(float(prob), 4)
        for label, prob in zip(CLASS_NAMES, probs)
    }


def get_top_info(probs: np.ndarray):
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])

    top1_label = CLASS_NAMES[top1_idx]
    top2_label = CLASS_NAMES[top2_idx]

    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx])

    margin = top1_prob - top2_prob

    return {
        "top1_idx": top1_idx,
        "top2_idx": top2_idx,
        "top1_label": top1_label,
        "top2_label": top2_label,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "margin": margin,
    }


def confidence_level(conf: float, margin: float) -> str:
    if conf >= 0.75 and margin >= 0.30:
        return "Yüksek"
    elif conf >= 0.50 and margin >= 0.15:
        return "Orta"
    else:
        return "Düşük"


def detect_and_crop_face(pil_image: Image.Image):
    image_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None, 0

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    cropped_face = image_np[y:y + h, x:x + w]
    cropped_pil = Image.fromarray(cropped_face)

    return cropped_pil, len(faces)


# -------------------------------------------------
# PREDICTION FUNCTIONS
# -------------------------------------------------
def predict_text(text: str) -> np.ndarray:
    cleaned = clean_text(text)

    inputs = text_tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # label encoder classes_: ['kizgin','korku','mutlu','surpriz','uzgun']
    prob_map = {label: float(prob) for label, prob in zip(label_encoder.classes_, probs)}

    # sabit fusion sırası: angry, fear, happy, sad, surprise
    ordered_probs = np.array([
        prob_map.get("kizgin", 0.0),
        prob_map.get("korku", 0.0),
        prob_map.get("mutlu", 0.0),
        prob_map.get("uzgun", 0.0),
        prob_map.get("surpriz", 0.0),
    ], dtype=float)

    ordered_probs = ordered_probs / ordered_probs.sum()
    return ordered_probs


def predict_image(pil_image: Image.Image):
    face_image, face_count = detect_and_crop_face(pil_image)

    if face_image is None:
        raise ValueError("Yüz tespit edilemedi.")

    image = image_transform(face_image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(image)
        probs = torch.softmax(outputs, dim=1)

    return probs.cpu().numpy()[0], face_image, face_count


# -------------------------------------------------
# FUSION
# -------------------------------------------------
def fuse_predictions(text_probs: np.ndarray, image_probs: np.ndarray):
    text_info = get_top_info(text_probs)
    image_info = get_top_info(image_probs)

    text_label = text_info["top1_label"]
    image_label = image_info["top1_label"]

    text_conf = text_info["top1_prob"]
    image_conf = image_info["top1_prob"]

    text_margin = text_info["margin"]
    image_margin = image_info["margin"]

    text_conf_level = confidence_level(text_conf, text_margin)
    image_conf_level = confidence_level(image_conf, image_margin)

    # aynı tahmin
    if text_label == image_label:
        final_probs = (text_probs + image_probs) / 2
        final_label = text_label
        final_sorted = np.sort(final_probs)
        final_conf = float(final_sorted[-1])
        final_margin = float(final_sorted[-1] - final_sorted[-2])

        return {
            "final_probs": final_probs,
            "final_label": final_label,
            "reason": "Metin ve görsel model aynı duygu tahminini verdi.",
            "text_conf": text_conf,
            "image_conf": image_conf,
            "text_conf_level": text_conf_level,
            "image_conf_level": image_conf_level,
            "final_conf": final_conf,
            "final_conf_level": confidence_level(final_conf, final_margin),
        }

    # text çok netse text seç
    if text_conf >= 0.80 and text_margin >= 0.25:
        return {
            "final_probs": text_probs,
            "final_label": text_label,
            "reason": "Metin modeli yüksek güvenle tahmin yaptığı için nihai karar metin modeline göre verildi.",
            "text_conf": text_conf,
            "image_conf": image_conf,
            "text_conf_level": text_conf_level,
            "image_conf_level": image_conf_level,
            "final_conf": text_conf,
            "final_conf_level": text_conf_level,
        }

    # image çok netse image seç
    if image_conf >= 0.80 and image_margin >= 0.25 and text_conf < 0.75:
        return {
            "final_probs": image_probs,
            "final_label": image_label,
            "reason": "Görsel model daha yüksek güvenle tahmin yaptığı için nihai karar görsel modele göre verildi.",
            "text_conf": text_conf,
            "image_conf": image_conf,
            "text_conf_level": text_conf_level,
            "image_conf_level": image_conf_level,
            "final_conf": image_conf,
            "final_conf_level": image_conf_level,
        }

    # dinamik ağırlık
    text_score = text_conf + text_margin
    image_score = image_conf + image_margin
    score_sum = text_score + image_score

    if score_sum == 0:
        dyn_text_weight = 0.6
        dyn_image_weight = 0.4
    else:
        dyn_text_weight = text_score / score_sum
        dyn_image_weight = image_score / score_sum

    # text biraz daha güçlü olduğu için hafif bias
    dyn_text_weight = dyn_text_weight * 0.60 + 0.10
    dyn_image_weight = dyn_image_weight * 0.40

    total_weight = dyn_text_weight + dyn_image_weight
    dyn_text_weight /= total_weight
    dyn_image_weight /= total_weight

    final_probs = dyn_text_weight * text_probs + dyn_image_weight * image_probs
    final_idx = int(np.argmax(final_probs))
    final_label = CLASS_NAMES[final_idx]

    final_sorted = np.sort(final_probs)
    final_conf = float(final_sorted[-1])
    final_margin = float(final_sorted[-1] - final_sorted[-2])
    final_conf_level = confidence_level(final_conf, final_margin)

    reason = (
        f"Metin modeli '{label_to_turkish(text_label)}', görsel model ise "
        f"'{label_to_turkish(image_label)}' tahmini verdi. "
        f"Model güvenleri dikkate alınarak dinamik ağırlıklı birleşim uygulandı "
        f"(text ağırlığı: {dyn_text_weight:.2f}, görsel ağırlığı: {dyn_image_weight:.2f})."
    )

    return {
        "final_probs": final_probs,
        "final_label": final_label,
        "reason": reason,
        "text_conf": text_conf,
        "image_conf": image_conf,
        "text_conf_level": text_conf_level,
        "image_conf_level": image_conf_level,
        "final_conf": final_conf,
        "final_conf_level": final_conf_level,
    }


# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="Çok Kipli Duygu Tanıma", layout="wide")

st.title("Çok Kipli Duygu Tanıma Sistemi")
st.write("Metin ve yüz görseli kullanarak duygu tahmini yapar.")

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

    try:
        image_probs, face_image, face_count = predict_image(pil_image)
    except Exception as e:
        st.error(f"Görsel analiz hatası: {e}")
        st.stop()

    text_pred = CLASS_NAMES[int(np.argmax(text_probs))]
    image_pred = CLASS_NAMES[int(np.argmax(image_probs))]

    fusion_result = fuse_predictions(text_probs, image_probs)

    final_probs = fusion_result["final_probs"]
    fusion_pred = fusion_result["final_label"]
    fusion_reason = fusion_result["reason"]

    text_conf = fusion_result["text_conf"]
    image_conf = fusion_result["image_conf"]

    text_conf_level = fusion_result["text_conf_level"]
    image_conf_level = fusion_result["image_conf_level"]

    final_conf = fusion_result["final_conf"]
    final_conf_level = fusion_result["final_conf_level"]

    st.subheader("Sonuçlar")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown("### Metin Modeli")
        st.write(f"**Tahmin:** {label_to_turkish(text_pred)}")
        st.write(f"**Güven Skoru:** {text_conf:.2f}")
        st.write(f"**Güven Seviyesi:** {text_conf_level}")
        st.json(probs_to_dict(text_probs))

    with result_col2:
        st.markdown("### Görsel Model")
        st.write(f"**Tahmin:** {label_to_turkish(image_pred)}")
        st.write(f"**Güven Skoru:** {image_conf:.2f}")
        st.write(f"**Güven Seviyesi:** {image_conf_level}")
        st.image(face_image, caption=f"Tespit edilen yüz (bulunan yüz sayısı: {face_count})", width=220)
        st.json(probs_to_dict(image_probs))

    with result_col3:
        st.markdown("### Fusion Sonucu")
        st.write(f"**Nihai Tahmin:** {label_to_turkish(fusion_pred)}")
        st.write(f"**Nihai Güven Skoru:** {final_conf:.2f}")
        st.write(f"**Nihai Güven Seviyesi:** {final_conf_level}")
        st.write(f"**Karar Nedeni:** {fusion_reason}")
        st.json(probs_to_dict(final_probs))

    if final_conf_level == "Düşük":
        st.warning(
            f"Nihai duygu tahmini: {label_to_turkish(fusion_pred)} "
            f"(güven seviyesi: {final_conf_level})"
        )
    else:
        st.success(
            f"Nihai duygu tahmini: {label_to_turkish(fusion_pred)} "
            f"(güven seviyesi: {final_conf_level})"
        )