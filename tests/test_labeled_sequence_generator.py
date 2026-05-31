import numpy as np

from src.sequence_generator import create_labeled_sequences


def test_create_labeled_sequences_shape():
    X = np.arange(20).reshape(20, 1)
    y = np.array([0, 1] * 10)

    X_seq, y_seq = create_labeled_sequences(
        X,
        y,
        window_size=5
    )

    assert X_seq.shape == (15, 5, 1)
    assert y_seq.shape == (15,)
