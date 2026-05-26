import numpy as np

from src.normalizer import normalize_data


def test_normalize_data_shapes():
    X_train = np.random.rand(60, 5)
    X_val = np.random.rand(20, 5)
    X_test = np.random.rand(20, 5)

    (
        X_train_norm,
        X_val_norm,
        X_test_norm
    ) = normalize_data(
        X_train,
        X_val,
        X_test
    )

    assert X_train_norm.shape == X_train.shape
    assert X_val_norm.shape == X_val.shape
    assert X_test_norm.shape == X_test.shape
