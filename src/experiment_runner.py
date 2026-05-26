import numpy as np

from src.metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_precision,
    calculate_recall
)

from src.runtime import measure_runtime


def run_experiment(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):
    runtime = measure_runtime(
        lambda: model.fit(
            X_train,
            y_train,
            verbose=0
        )
    )

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predictions = (
        predictions > 0.5
    ).astype(int).flatten()

    accuracy = calculate_accuracy(
        y_test,
        predictions
    )

    f1_score = calculate_f1_score(
        y_test,
        predictions
    )

    precision = calculate_precision(
    y_test,
    predictions
   )

    recall = calculate_recall(
    y_test,
    predictions
   )

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "runtime": runtime,
        "precision": precision,
        "recall": recall
    }
