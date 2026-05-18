import numpy as np

from src.metrics import calculate_confusion_matrix


def test_confusion_matrix_shape():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred
    )

    assert matrix.shape == (2, 2)
