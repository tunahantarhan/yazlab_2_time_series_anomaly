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
from src.models.lstm_model import build_lstm_model
from src.trainer import train_model


print("BATADAL yukleniyor...")

df = pd.read_csv(
    "data/batadal/test_dataset.csv"
)

df.columns = df.columns.str.strip()

df = df.ffill()
df = df.bfill()

dataset = prepare_dataset(
    df,
    target_column="ATT_FLAG",
    columns_to_drop=["DATETIME"]
)

X_train_seq, y_train_seq = create_labeled_sequences(
    dataset["X_train"],
    dataset["y_train"],
    window_size=5
)

X_val_seq, y_val_seq = create_labeled_sequences(
    dataset["X_val"],
    dataset["y_val"],
    window_size=5
)

model = build_lstm_model(
    input_shape=(
        X_train_seq.shape[1],
        X_train_seq.shape[2]
    )
)

start_train = time.time()

history = train_model(
    model,
    X_train_seq,
    y_train_seq,
    X_val_seq,
    y_val_seq
)

training_time = time.time() - start_train

print("Egitim tamamlandi.")

start_inference = time.time()

predictions = model.predict(
    X_val_seq,
    verbose=0
)

inference_time = time.time() - start_inference

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
print("=== RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Training Time : {training_time:.4f} sn")
print(f"Inference Time: {inference_time:.4f} sn")
print()
print("Confusion Matrix")
print(cm)
