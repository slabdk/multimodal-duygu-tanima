import sys
import pandas as pd
from huggingface_hub import hf_hub_download
from src.utils.text_preprocessing import clean_text

print("Python yolu:", sys.executable)
print("-" * 50)

# CSV dosyasını Hugging Face'ten indir
file_path = hf_hub_download(
    repo_id="anilguven/turkish_tweet_emotion_dataset",
    filename="Turkish_Tweet_Dataset.csv",
    repo_type="dataset"
)

print("Dosya yolu:", file_path)
print("-" * 50)

# CSV'yi kontrollü oku
df = pd.read_csv(
    file_path,
    sep=",",
    header=None,
    names=["text", "label"],
    engine="python",
    quotechar='"',
    on_bad_lines="skip"
)

print("Kolonlar:")
print(df.columns)
print("-" * 50)

print("İlk 5 satır:")
print(df.head())
print("-" * 50)

print("Veri seti boyutu:", df.shape)
print("-" * 50)

print("Eksik değer sayısı:")
print(df.isnull().sum())

print("Duygu dağılımı:")
print(df["label"].value_counts())
print("-" * 50)


df = df.dropna(subset=["text"]).copy()
df["clean_text"] = df["text"].apply(clean_text)
print(df[["text", "clean_text"]].head())
print("Yeni boyut:", df.shape)
