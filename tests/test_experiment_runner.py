import numpy as np

from src.experiment_runner import run_experiment


class DummyModel:
    def fit(self, *args, **kwargs):
        return None

    def predict(self, X, verbose=0):
        return np.ones((len(X), 1))


def test_run_experiment_returns_metrics():
    X_train = np.random.rand(20, 5, 1)
    y_train = np.random.randint(0, 2, 20)

    X_test = np.random.rand(10, 5, 1)
    y_test = np.random.randint(0, 2, 10)

    model = DummyModel()

    results = run_experiment(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    assert "accuracy" in results
    assert "runtime" in results
    assert "f1_score" in results
