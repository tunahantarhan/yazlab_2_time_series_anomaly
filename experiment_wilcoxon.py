# -*- coding: utf-8 -*-

from src.statistical_tests import run_wilcoxon_test


gru_scores = [
    0.3725,
    0.3212,
    0.6133,
    0.3360,
    0.1562
]

cnn_scores = [
    0.5161,
    0.4329,
    0.6341,
    0.4386,
    0.2484
]

statistic, p_value = run_wilcoxon_test(
    gru_scores,
    cnn_scores
)

print("=== WILCOXON TEST RESULTS ===")
print(f"Statistic: {statistic:.4f}")
print(f"P-value  : {p_value:.4f}")

if p_value < 0.05:
    print("Result: Modeller arasinda istatistiksel olarak anlamli fark vardir.")
else:
    print("Result: Modeller arasinda istatistiksel olarak anlamli fark yoktur.")
