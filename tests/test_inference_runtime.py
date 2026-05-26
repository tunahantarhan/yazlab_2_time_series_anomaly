import numpy as np

from src.inference import measure_inference_time


class DummyModel:
    def predict(self, X, verbose=0):
        return np.zeros((len(X), 1))


def test_inference_runtime_returns_positive_value():
    X = np.random.rand(20, 5, 1)

    model = DummyModel()

    runtime = measure_inference_time(
        model,
        X
    )

    assert runtime >= 0
