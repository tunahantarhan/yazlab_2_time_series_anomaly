import numpy as np

from src.pca import apply_pca


def test_pca_output_shape():
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(20, 5)

    X_train_pca, X_test_pca = apply_pca(
        X_train,
        X_test
    )

    assert X_train_pca.shape == (100, 1)
    assert X_test_pca.shape == (20, 1)
