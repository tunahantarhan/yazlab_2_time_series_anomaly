# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

from sklearn.metrics import f1_score

from src.seed import set_seed
from src.preprocessing_pipeline import prepare_dataset
from src.sequence_generator import create_labeled_sequences
from src.models.lstm_model import build_lstm_model
from src.trainer import train_model


SEEDS = [42, 123, 2026, 7, 999]


def load_swat():

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

    return (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq
    )


def run_lstm(seed):

    set_seed(seed)

    (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq
    ) = load_swat()

    model = build_lstm_model(
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

    training_time = (
        time.time() - start_time
    )

    predictions = model.predict(
        X_val_seq,
        verbose=0
    )

    predictions = (
        predictions > 0.5
    ).astype(int).flatten()

    f1 = f1_score(
        y_val_seq,
        predictions,
        zero_division=0
    )

    return f1, training_time


if __name__ == "__main__":

    print("\n=== LSTM SEED RESULTS ===")

    scores = []
    times = []

    for seed in SEEDS:

        f1, training_time = run_lstm(seed)

        scores.append(f1)
        times.append(training_time)

        print(
            f"Seed {seed} | "
            f"F1: {f1:.4f} | "
            f"Training Time: {training_time:.4f} sn"
        )

    print()
    print(f"Mean F1: {np.mean(scores):.4f}")
    print(f"Std F1 : {np.std(scores):.4f}")
    print(
        f"Mean Training Time: "
        f"{np.mean(times):.4f} sn"
    )
