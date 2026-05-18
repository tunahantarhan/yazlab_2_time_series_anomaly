import numpy as np

from src.models.gru_model import build_gru_model
from src.trainer import train_model


def test_train_model_returns_history():
    X = np.random.rand(50, 5, 1)
    y = np.random.randint(0, 2, 50)

    model = build_gru_model(
        input_shape=(5, 1)
    )

    history = train_model(
        model,
        X,
        y
    )

    assert history is not None
