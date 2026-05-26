from scipy.stats import wilcoxon


def run_wilcoxon_test(
    scores_model_1,
    scores_model_2
):
    statistic, p_value = wilcoxon(
        scores_model_1,
        scores_model_2
    )

    return statistic, p_value
