import numpy as np


def summarize_metric(results, metric_name):
    values = [
        result[metric_name]
        for result in results
    ]

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values))
    }
