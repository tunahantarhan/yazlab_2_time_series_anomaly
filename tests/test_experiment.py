from src.experiment import run_experiment


def test_experiment_runs():
    rows, cols = run_experiment(
        "data/swat/sample.csv"
    )

    assert rows > 0
    assert cols > 0
