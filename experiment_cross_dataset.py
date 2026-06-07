# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.sequence_generator import create_labeled_sequences
from src.trainer import train_model
from src.models.gru_model import build_gru_model
from src.models.cnn_model import build_cnn_model
from src.models.lstm_model import build_lstm_model


WINDOW_SIZE = 5


def load_swat():
    df = pd.read_csv("data/swat/merged.csv")
    df.columns = df.columns.str.strip()
    df = df.ffill().bfill()

    df["Normal/Attack"] = (
        df["Normal/Attack"]
        .map({
            "Normal": 0,
            "Attack": 1
        })
    )

    normal_df = df[df["Normal/Attack"] == 0].head(25000)
    attack_df = df[df["Normal/Attack"] == 1].head(25000)

    df = pd.concat(
        [normal_df, attack_df],
        ignore_index=True
    )

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    y = df["Normal/Attack"].values

    X = df.drop(
        columns=["Normal/Attack"]
    )

    sensor_column = X.columns[0]

    X = X[sensor_column].values.reshape(-1, 1)

    return X, y


def load_batadal():
    df = pd.read_csv("data/batadal/test_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.ffill().bfill()

    if "DATETIME" in df.columns:
        df = df.drop(columns=["DATETIME"])

    y = df["ATT_FLAG"].values.astype(int)

    X = df.drop(
        columns=["ATT_FLAG"]
    )

    sensor_column = X.columns[0]

    X = X[sensor_column].values.reshape(-1, 1)

    return X, y


def split_train_val_test(X, y):
    train_end = int(len(X) * 0.60)
    val_end = int(len(X) * 0.80)

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:]
    }


def build_model(model_name, input_shape):
    if model_name == "GRU":
        return build_gru_model(input_shape)

    if model_name == "1D-CNN":
        return build_cnn_model(input_shape)

    if model_name == "LSTM":
        return build_lstm_model(input_shape)

    raise ValueError("Bilinmeyen model tipi")


def prepare_cross_data(train_dataset, test_dataset):
    train_split = split_train_val_test(
        train_dataset[0],
        train_dataset[1]
    )

    test_split = split_train_val_test(
        test_dataset[0],
        test_dataset[1]
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_split["X_train"]
    )

    X_val = scaler.transform(
        train_split["X_val"]
    )

    X_test = scaler.transform(
        test_split["X_test"]
    )

    X_train_seq, y_train_seq = create_labeled_sequences(
        X_train,
        train_split["y_train"],
        WINDOW_SIZE
    )

    X_val_seq, y_val_seq = create_labeled_sequences(
        X_val,
        train_split["y_val"],
        WINDOW_SIZE
    )

    X_test_seq, y_test_seq = create_labeled_sequences(
        X_test,
        test_split["y_test"],
        WINDOW_SIZE
    )

    return (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        X_test_seq,
        y_test_seq
    )


def run_cross_dataset_experiment(
    model_name,
    train_name,
    train_dataset,
    test_name,
    test_dataset
):
    print()
    print(
        f"=== {model_name} | Train: {train_name} -> Test: {test_name} ==="
    )

    (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        X_test_seq,
        y_test_seq
    ) = prepare_cross_data(
        train_dataset,
        test_dataset
    )

    model = build_model(
        model_name,
        input_shape=(
            X_train_seq.shape[1],
            X_train_seq.shape[2]
        )
    )

    start_time = time.time()

    train_model(
        model,
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq
    )

    training_time = time.time() - start_time

    predictions = model.predict(
        X_test_seq,
        verbose=0
    )

    predictions = (
        predictions > 0.5
    ).astype(int).flatten()

    accuracy = accuracy_score(
        y_test_seq,
        predictions
    )

    precision = precision_score(
        y_test_seq,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_seq,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_seq,
        predictions,
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Training Time: {training_time:.4f} sn")

    return {
        "model": model_name,
        "train": train_name,
        "test": test_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time
    }


if __name__ == "__main__":
    print("Cross-dataset deneyleri baslatiliyor...")

    swat_dataset = load_swat()
    batadal_dataset = load_batadal()

    models = [
        "LSTM",
        "GRU",
        "1D-CNN"
    ]

    results = []

    for model_name in models:
        results.append(
            run_cross_dataset_experiment(
                model_name,
                "SWAT",
                swat_dataset,
                "BATADAL",
                batadal_dataset
            )
        )

        results.append(
            run_cross_dataset_experiment(
                model_name,
                "BATADAL",
                batadal_dataset,
                "SWAT",
                swat_dataset
            )
        )

    print()
    print("=== CROSS-DATASET SUMMARY ===")

    for result in results:
        print(
            f"{result['model']} | "
            f"Train: {result['train']} -> Test: {result['test']} | "
            f"F1: {result['f1_score']:.4f}"
        )
