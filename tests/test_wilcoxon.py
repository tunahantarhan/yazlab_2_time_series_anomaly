import numpy as np

from src.statistical_tests import run_wilcoxon_test


def test_run_wilcoxon_test_returns_pvalue():
    scores_model_1 = np.array([
        0.80,
        0.82,
        0.81,
        0.79,
        0.83
    ])

    scores_model_2 = np.array([
        0.75,
        0.74,
        0.76,
        0.73,
        0.77
    ])

    statistic, p_value = run_wilcoxon_test(
        scores_model_1,
        scores_model_2
    )

    assert p_value >= 0
