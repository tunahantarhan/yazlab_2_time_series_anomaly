from src.models.gru_model import build_gru_model


def test_gru_model_creation():
    model = build_gru_model(
        input_shape=(10, 1)
    )

    assert model is not None

def test_gru_model_has_layers():
    model = build_gru_model(
        input_shape=(10, 1)
    )

    assert len(model.layers) > 0

def test_model_contains_gru_layer():
    model = build_gru_model(
        input_shape=(10, 1)
    )

    layer_names = [
        layer.__class__.__name__
        for layer in model.layers
    ]

    assert "GRU" in layer_names
