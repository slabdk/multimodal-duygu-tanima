import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.utils.text_preprocessing import clean_text


# Veri setini indir
file_path = hf_hub_download(
    repo_id="anilguven/turkish_tweet_emotion_dataset",
    filename="Turkish_Tweet_Dataset.csv",
    repo_type="dataset"
)

# CSV oku
df = pd.read_csv(
    file_path,
    sep=",",
    header=None,
    names=["text", "label"],
    engine="python",
    quotechar='"',
    on_bad_lines="skip"
)

# eksik verileri kaldır
df = df.dropna(subset=["text"]).copy()

# temiz metin
df["clean_text"] = df["text"].apply(clean_text)

print("Veri boyutu:", df.shape)

# X ve y
X = df["clean_text"]
y = df["label"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

# tahmin
y_pred = model.predict(X_test_tfidf)

# sonuç
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))



print("Sınıf sırası:", model.classes_)



cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Text Model Confusion Matrix")
plt.show()