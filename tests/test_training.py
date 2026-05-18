import numpy as np

from src.models.gru_model import build_gru_model


def test_gru_training_runs():
    X = np.random.rand(50, 5, 1)
    y = np.random.randint(0, 2, 50)

    model = build_gru_model(
        input_shape=(5, 1)
    )

    history = model.fit(
        X,
        y,
        epochs=1,
        batch_size=8,
        verbose=0
    )

    assert history is not None
