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
# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="Çok Kipli Duygu Tanıma", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #aaa;
    margin-bottom: 25px;
}

.card {
    padding: 18px;
    border-radius: 12px;
    background-color: #1e1e1e;
    border: 1px solid #333;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main-title">Multimodal Duygu Analizi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Metin ve/veya yüz görseli ile duygu tahmini yapabilirsiniz</div>', unsafe_allow_html=True)

# ---------- INPUT ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Metin Girişi")
    user_text = st.text_area(
        "",
        placeholder="Bugün kendimi çok kötü hissediyorum..."
    )

with col2:
    st.markdown("### Görsel Yükleme")
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Yüklenen Görsel", width=200)

# ---------- BUTTON ----------
analyze_clicked = st.button("Analiz Et")

# ---------- HELPER UI ----------
def show_confidence(value):
    st.progress(float(value))
    st.write(f"Güven Skoru: {value:.2f}")

def result_card(title, label, confidence, probs):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.write(f"**Tahmin:** {label}")
    show_confidence(confidence)

    with st.expander("Detaylı Olasılıklar"):
        st.json(probs)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ANALYZE ----------
if analyze_clicked:

    has_text = bool(user_text.strip())
    has_image = uploaded_image is not None

    if not has_text and not has_image:
        st.error("Lütfen en az bir veri gir.")
        st.stop()

    # TEXT ONLY
    if has_text and not has_image:
        st.info("Mod: Sadece Metin")

        text_probs = predict_text(user_text)
        text_pred = CLASS_NAMES[int(np.argmax(text_probs))]
        text_conf = float(np.max(text_probs))

        result_card(
            "Metin Modeli",
            label_to_turkish(text_pred),
            text_conf,
            probs_to_dict(text_probs)
        )

    # IMAGE ONLY
    elif has_image and not has_text:
        st.info("Mod: Sadece Görsel")

        pil_image = Image.open(uploaded_image).convert("RGB")

        image_probs, face_image, face_count = predict_image(pil_image)

        image_pred = CLASS_NAMES[int(np.argmax(image_probs))]
        image_conf = float(np.max(image_probs))

        st.image(face_image, caption=f"Yüz sayısı: {face_count}", width=220)

        result_card(
            "Görsel Model",
            label_to_turkish(image_pred),
            image_conf,
            probs_to_dict(image_probs)
        )

    # MULTIMODAL
    else:
        st.info("Mod: Multimodal")

        pil_image = Image.open(uploaded_image).convert("RGB")

        text_probs = predict_text(user_text)
        image_probs, face_image, face_count = predict_image(pil_image)

        fusion_result = fuse_predictions(text_probs, image_probs)

        text_pred = CLASS_NAMES[int(np.argmax(text_probs))]
        image_pred = CLASS_NAMES[int(np.argmax(image_probs))]
        fusion_pred = fusion_result["final_label"]

        st.image(face_image, caption=f"Yüz sayısı: {face_count}", width=220)

        col1, col2, col3 = st.columns(3)

        with col1:
            result_card(
                "Metin Modeli",
                label_to_turkish(text_pred),
                fusion_result["text_conf"],
                probs_to_dict(text_probs)
            )

        with col2:
            result_card(
                "Görsel Model",
                label_to_turkish(image_pred),
                fusion_result["image_conf"],
                probs_to_dict(image_probs)
            )

        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Fusion Sonucu")

            st.write(f"**Nihai Tahmin:** {label_to_turkish(fusion_pred)}")
            show_confidence(fusion_result["final_conf"])

            st.write(f"**Karar Nedeni:** {fusion_result['reason']}")

            with st.expander("Fusion Detayları"):
                st.json(probs_to_dict(fusion_result["final_probs"]))

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------
        # FINAL RESULT BANNER
        # ---------------------------

        final_label_tr = label_to_turkish(fusion_pred)
        final_conf = fusion_result["final_conf"]
        final_conf_level = fusion_result["final_conf_level"]

        st.markdown("---")  # ayırıcı çizgi

        if final_conf_level == "Düşük":
            st.warning(
                f"🔴 Nihai Duygu: {final_label_tr} | Güven Seviyesi: {final_conf_level} ({final_conf:.2f})"
            )
        else:
            st.success(
                f"🟢 Nihai Duygu: {final_label_tr} | Güven Seviyesi: {final_conf_level} ({final_conf:.2f})"
            )