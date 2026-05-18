import numpy as np

from src.metrics import (
    calculate_precision,
    calculate_recall
)


def test_precision_calculation():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision = calculate_precision(
        y_true,
        y_pred
    )

    assert round(precision, 2) == 1.00


def test_recall_calculation():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    recall = calculate_recall(
        y_true,
        y_pred
    )

    assert round(recall, 2) == 0.67
