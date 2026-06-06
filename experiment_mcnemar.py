# -*- coding: utf-8 -*-

import numpy as np

from src.statistical_tests import run_mcnemar_test


y_true = np.array([
    0, 0, 0, 0, 1, 1, 1, 1, 1, 0
])

gru_predictions = np.array([
    0, 0, 1, 0, 1, 1, 0, 1, 0, 0
])

cnn_predictions = np.array([
    0, 1, 0, 0, 1, 0, 1, 1, 1, 0
])

statistic, p_value = run_mcnemar_test(
    y_true,
    gru_predictions,
    cnn_predictions
)

print("=== MCNEMAR TEST RESULTS ===")
print(f"Statistic: {statistic:.4f}")
print(f"P-value  : {p_value:.4f}")

if p_value < 0.05:
    print(
        "Result: Modeller arasinda istatistiksel olarak anlamli fark vardir."
    )
else:
    print(
        "Result: Modeller arasinda istatistiksel olarak anlamli fark yoktur."
    )
