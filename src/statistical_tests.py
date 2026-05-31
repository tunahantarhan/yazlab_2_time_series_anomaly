from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np


def run_wilcoxon_test(
    scores_model_1,
    scores_model_2
):
    statistic, p_value = wilcoxon(
        scores_model_1,
        scores_model_2
    )

    return statistic, p_value


def run_mcnemar_test(
    y_true,
    model_1_predictions,
    model_2_predictions
):
    table = np.zeros(
        (2, 2),
        dtype=int
    )

    for true, pred1, pred2 in zip(
        y_true,
        model_1_predictions,
        model_2_predictions
    ):
        correct_1 = pred1 == true
        correct_2 = pred2 == true

        if correct_1 and correct_2:
            table[0, 0] += 1
        elif correct_1 and not correct_2:
            table[0, 1] += 1
        elif not correct_1 and correct_2:
            table[1, 0] += 1
        else:
            table[1, 1] += 1

    result = mcnemar(
        table,
        exact=False,
        correction=True
    )

    return result.statistic, result.pvalue
