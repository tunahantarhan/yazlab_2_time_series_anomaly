# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from src.models.gru_model import build_gru_model
from src.models.cnn_model import build_cnn_model
from src.trainer import train_model


print("SWAT K-Fold baslatiliyor...")


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

if "Timestamp" in df.columns:
    df = df.drop(
        columns=["Timestamp"]
    )

X = df.drop(
    columns=["Normal/Attack"]
).values

y = df["Normal/Attack"].values

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

gru_scores = []
cnn_scores = []

fold = 1

for train_idx, test_idx in kfold.split(X):

    print(f"\nFold {fold}")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    gru_model = build_gru_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    train_model(
        gru_model,
        X_train,
        y_train,
        epochs=10
    )

    gru_pred = (
        gru_model.predict(
            X_test,
            verbose=0
        ) > 0.5
    ).astype(int)

    gru_f1 = f1_score(
        y_test,
        gru_pred,
        zero_division=0
    )

    gru_scores.append(gru_f1)

    cnn_model = build_cnn_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    train_model(
        cnn_model,
        X_train,
        y_train,
        epochs=10
    )

    cnn_pred = (
        cnn_model.predict(
            X_test,
            verbose=0
        ) > 0.5
    ).astype(int)

    cnn_f1 = f1_score(
        y_test,
        cnn_pred,
        zero_division=0
    )

    cnn_scores.append(cnn_f1)

    print(f"GRU F1 : {gru_f1:.4f}")
    print(f"CNN F1 : {cnn_f1:.4f}")

    fold += 1

print("\n=== K-FOLD RESULTS ===")

print(
    f"GRU Mean F1 : {np.mean(gru_scores):.4f}"
)

print(
    f"GRU Std F1  : {np.std(gru_scores):.4f}"
)

print(
    f"CNN Mean F1 : {np.mean(cnn_scores):.4f}"
)

print(
    f"CNN Std F1  : {np.std(cnn_scores):.4f}"
)
