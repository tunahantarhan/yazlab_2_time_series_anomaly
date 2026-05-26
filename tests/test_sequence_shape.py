import numpy as np

from src.sequence_generator import create_sequences


def test_sequence_generator_returns_3d_tensor():
    data = np.arange(20)

    X, y = create_sequences(
        data,
        window_size=5
    )

    assert len(X.shape) == 3
    assert X.shape[1] == 5
    assert X.shape[2] == 1
