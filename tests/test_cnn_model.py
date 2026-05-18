from src.models.cnn_model import build_cnn_model


def test_cnn_model_creation():
    model = build_cnn_model(
        input_shape=(10, 1)
    )

    assert model is not None
