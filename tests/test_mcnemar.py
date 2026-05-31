import numpy as np

from src.statistical_tests import run_mcnemar_test


def test_run_mcnemar_test_returns_pvalue():
    y_true = np.array([
        1, 0, 1, 0,
        1, 0, 1, 0
    ])

    model_1_predictions = np.array([
        1, 0, 1, 0,
        1, 0, 0, 0
    ])

    model_2_predictions = np.array([
        1, 1, 1, 0,
        0, 0, 1, 0
    ])

    statistic, p_value = run_mcnemar_test(
        y_true,
        model_1_predictions,
        model_2_predictions
    )

    assert p_value >= 0
