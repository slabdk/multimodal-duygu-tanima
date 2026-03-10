from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from src.utils.text_preprocessing import clean_text

# -------------------------------------------------
# PATH
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# -------------------------------------------------
# DATASET
# -------------------------------------------------
file_path = hf_hub_download(
    repo_id="anilguven/turkish_tweet_emotion_dataset",
    filename="Turkish_Tweet_Dataset.csv",
    repo_type="dataset",
)

df = pd.read_csv(
    file_path,
    sep=",",
    header=None,
    names=["text", "label"],
    engine="python",
    quotechar='"',
    on_bad_lines="skip",
)

# Eksik text verilerini kaldır
df = df.dropna(subset=["text"]).copy()

# Temizleme
df["clean_text"] = df["text"].apply(clean_text)

print("Veri boyutu:", df.shape)
print("-" * 50)

# -------------------------------------------------
# TRAIN / TEST
# -------------------------------------------------
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("-" * 50)

# -------------------------------------------------
# TF-IDF
# -------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)
print("-" * 50)

# -------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_tfidf, y_train)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("-" * 50)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("-" * 50)

print("Sınıf sırası:", model.classes_)
print("-" * 50)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_,
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Text Model Confusion Matrix - Random Forest")
plt.show()

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
joblib.dump(model, BASE_DIR / "models" / "text_model.pkl")
joblib.dump(vectorizer, BASE_DIR / "models" / "tfidf_vectorizer.pkl")

print("Random Forest text modeli kaydedildi.")
print("TF-IDF vectorizer kaydedildi.")