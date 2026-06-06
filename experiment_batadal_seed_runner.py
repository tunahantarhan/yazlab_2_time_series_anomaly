# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

from sklearn.metrics import f1_score

from src.seed import set_seed
from src.preprocessing_pipeline import prepare_dataset
from src.sequence_generator import create_labeled_sequences

from src.models.gru_model import build_gru_model
from src.models.cnn_model import build_cnn_model

from src.trainer import train_model


SEEDS = [42, 123, 2026, 7, 999]


def load_batadal():

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

    return (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq
    )


def run_gru(seed):

    set_seed(seed)

    (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq
    ) = load_batadal()

    model = build_gru_model(
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


def run_cnn(seed):

    set_seed(seed)

    (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq
    ) = load_batadal()

    model = build_cnn_model(
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

    print("\n=== BATADAL GRU SEED RESULTS ===")

    gru_scores = []
    gru_times = []

    for seed in SEEDS:

        f1, training_time = run_gru(seed)

        gru_scores.append(f1)
        gru_times.append(training_time)

        print(
            f"Seed {seed} | "
            f"F1: {f1:.4f} | "
            f"Training Time: {training_time:.4f} sn"
        )

    print()
    print(f"Mean F1: {np.mean(gru_scores):.4f}")
    print(f"Std F1 : {np.std(gru_scores):.4f}")
    print(
        f"Mean Training Time: "
        f"{np.mean(gru_times):.4f} sn"
    )

    print("\n=== BATADAL CNN SEED RESULTS ===")

    cnn_scores = []
    cnn_times = []

    for seed in SEEDS:

        f1, training_time = run_cnn(seed)

        cnn_scores.append(f1)
        cnn_times.append(training_time)

        print(
            f"Seed {seed} | "
            f"F1: {f1:.4f} | "
            f"Training Time: {training_time:.4f} sn"
        )

    print()
    print(f"Mean F1: {np.mean(cnn_scores):.4f}")
    print(f"Std F1 : {np.std(cnn_scores):.4f}")
    print(
        f"Mean Training Time: "
        f"{np.mean(cnn_times):.4f} sn"
    )
