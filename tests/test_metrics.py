import numpy as np

from src.metrics import calculate_accuracy


def test_accuracy_calculation():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    accuracy = calculate_accuracy(
        y_true,
        y_pred
    )

    assert accuracy == 0.75
