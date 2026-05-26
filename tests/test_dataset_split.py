import numpy as np

from src.dataset_split import split_dataset


def test_dataset_split_ratios():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)

    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20
