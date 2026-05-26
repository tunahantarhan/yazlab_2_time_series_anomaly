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


def test_train_model_accepts_validation_data():
    X_train = np.random.rand(40, 5, 1)
    y_train = np.random.randint(0, 2, 40)

    X_val = np.random.rand(10, 5, 1)
    y_val = np.random.randint(0, 2, 10)

    model = build_gru_model(
        input_shape=(5, 1)
    )

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=1,
        batch_size=8,
        patience=1
    )

    assert history is not None
    assert "loss" in history.history
