import numpy as np

from src.noise import add_gaussian_noise


def test_noise_changes_data():
    data = np.ones((10, 5))

    noisy_data = add_gaussian_noise(
        data,
        mean=0,
        std=0.1
    )

    assert not np.array_equal(
        data,
        noisy_data
    )
