import numpy as np

from src.normalizer import normalize_data


def test_normalizer_fits_only_train_data():
    X_train = np.array([
        [1.0],
        [2.0],
        [3.0]
    ])

    X_val = np.array([
        [100.0]
    ])

    X_test = np.array([
        [200.0]
    ])

    X_train_norm, X_val_norm, X_test_norm = normalize_data(
        X_train,
        X_val,
        X_test
    )

    assert round(X_train_norm.mean(), 6) == 0.0
    assert X_val_norm[0][0] > 10
    assert X_test_norm[0][0] > 10
