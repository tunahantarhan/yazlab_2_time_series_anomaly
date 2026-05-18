import numpy as np

from src.sequence_generator import create_sequences


def test_sequence_generation_shape():
    data = np.arange(20)

    X, y = create_sequences(
        data,
        window_size=5
    )

    assert X.shape[1] == 5
