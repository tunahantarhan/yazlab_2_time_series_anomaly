import numpy as np

from src.metrics import calculate_accuracy
from src.runtime import measure_runtime


def run_experiment(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):
    runtime = measure_runtime(
        model,
        X_train,
        y_train
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

    return {
        "accuracy": accuracy,
        "runtime": runtime
    }
