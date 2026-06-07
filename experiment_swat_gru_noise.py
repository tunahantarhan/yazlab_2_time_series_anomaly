# -*- coding: utf-8 -*-

import pandas as pd
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.preprocessing_pipeline import prepare_dataset
from src.sequence_generator import create_labeled_sequences
from src.models.gru_model import build_gru_model
from src.trainer import train_model
from src.noise import add_gaussian_noise


print("SWAT noisy GRU yukleniyor...")

df = pd.read_csv(
    "data/swat/merged.csv"
)

df.columns = df.columns.str.strip()

df = df.ffill()
df = df.bfill()

df["Normal/Attack"] = (
    df["Normal/Attack"]
    .map({
        "Normal": 0,
        "Attack": 1
    })
)

normal_df = df[
    df["Normal/Attack"] == 0
].head(25000)

attack_df = df[
    df["Normal/Attack"] == 1
].head(25000)

df = pd.concat(
    [normal_df, attack_df],
    ignore_index=True
)

dataset = prepare_dataset(
    df,
    target_column="Normal/Attack",
    columns_to_drop=["Timestamp"]
)

dataset["X_train"] = add_gaussian_noise(
    dataset["X_train"],
    std=0.1
)

dataset["X_val"] = add_gaussian_noise(
    dataset["X_val"],
    std=0.1
)

X_train_seq, y_train_seq = create_labeled_sequences(
    dataset["X_train"],
    dataset["y_train"],
    window_size=4
)

X_val_seq, y_val_seq = create_labeled_sequences(
    dataset["X_val"],
    dataset["y_val"],
    window_size=4
)

model = build_gru_model(
    input_shape=(
        X_train_seq.shape[1],
        X_train_seq.shape[2]
    )
)

train_start = time.time()

history = train_model(
    model,
    X_train_seq,
    y_train_seq,
    X_val_seq,
    y_val_seq
)

train_end = time.time()
training_time = train_end - train_start

print("Egitim tamamlandi.")

inference_start = time.time()

predictions = model.predict(
    X_val_seq,
    verbose=0
)

inference_end = time.time()
inference_time = inference_end - inference_start

predictions = (
    predictions > 0.5
).astype(int).flatten()

accuracy = accuracy_score(
    y_val_seq,
    predictions
)

precision = precision_score(
    y_val_seq,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val_seq,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_val_seq,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_val_seq,
    predictions
)

print()
print("=== NOISY RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Training Time : {training_time:.4f} sn")
print(f"Inference Time: {inference_time:.4f} sn")
print()
print("Confusion Matrix")
print(cm)
