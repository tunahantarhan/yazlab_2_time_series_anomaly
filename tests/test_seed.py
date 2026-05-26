import numpy as np

from src.seed import set_seed


def test_seed_reproducibility():
    set_seed(42)
    first = np.random.rand(5)

    set_seed(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)
