import numpy as np

from src.metrics import calculate_f1_score


def test_f1_score_calculation():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    f1 = calculate_f1_score(
        y_true,
        y_pred
    )

    assert round(f1, 2) == 0.80
