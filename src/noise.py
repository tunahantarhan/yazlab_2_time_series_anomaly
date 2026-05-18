import numpy as np


def add_gaussian_noise(data, mean=0, std=0.1):
    noise = np.random.normal(
        mean,
        std,
        data.shape
    )

    return data + noise
