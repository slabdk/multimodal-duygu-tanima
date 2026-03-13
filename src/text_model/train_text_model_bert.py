from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from src.utils.text_preprocessing import clean_text


# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "models" / "berturk_emotion_model"

MODEL_NAME = "dbmdz/bert-base-turkish-cased"


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

df = df.dropna(subset=["text"]).copy()
df["clean_text"] = df["text"].apply(clean_text)

print("Veri boyutu:", df.shape)
print(df["label"].value_counts())
print("-" * 50)


# -------------------------------------------------
# TRAIN / VALID / TEST SPLIT
# -------------------------------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)
print("-" * 50)


# -------------------------------------------------
# LABEL ENCODING
# -------------------------------------------------
label_encoder = LabelEncoder()

train_df["label_id"] = label_encoder.fit_transform(train_df["label"])
val_df["label_id"] = label_encoder.transform(val_df["label"])
test_df["label_id"] = label_encoder.transform(test_df["label"])

id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in id2label.items()}

print("Label mapping:", id2label)
print("-" * 50)


# -------------------------------------------------
# HF DATASETS
# -------------------------------------------------
train_ds = Dataset.from_pandas(train_df[["clean_text", "label_id"]].rename(
    columns={"clean_text": "text", "label_id": "label"}
))
val_ds = Dataset.from_pandas(val_df[["clean_text", "label_id"]].rename(
    columns={"clean_text": "text", "label_id": "label"}
))
test_ds = Dataset.from_pandas(test_df[["clean_text", "label_id"]].rename(
    columns={"clean_text": "text", "label_id": "label"}
))


# -------------------------------------------------
# TOKENIZER
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )


train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_),
    id2label=id2label,
    label2id=label2id,
)


# -------------------------------------------------
# METRICS
# -------------------------------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


# -------------------------------------------------
# TRAINING ARGS
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)


# -------------------------------------------------
# TRAINER
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# -------------------------------------------------
# TRAIN
# -------------------------------------------------
trainer.train()


# -------------------------------------------------
# EVALUATE ON TEST
# -------------------------------------------------
pred_output = trainer.predict(test_ds)

y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)

test_accuracy = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, average="macro")

print("Test Accuracy:", test_accuracy)
print("Test F1 Macro:", test_f1)
print("-" * 50)

target_names = [id2label[i] for i in range(len(id2label))]

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=target_names))
print("-" * 50)


# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(target_names)))
ax.set_yticks(range(len(target_names)))
ax.set_xticklabels(target_names, rotation=45)
ax.set_yticklabels(target_names)

for i in range(len(target_names)):
    for j in range(len(target_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

ax.set_title("BERTurk Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.show()


# -------------------------------------------------
# SAVE
# -------------------------------------------------
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

# Label encoder da kaydedelim
import joblib
joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.pkl")

print(f"Model kaydedildi: {OUTPUT_DIR}")