import numpy as np

from src.cross_validation import create_kfold_splits


def test_create_kfold_splits():
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)

    splits = create_kfold_splits(
        X,
        y,
        n_splits=5
    )

    assert len(splits) == 5
