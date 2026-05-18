import time

from src.runtime import measure_runtime


def dummy_function():
    time.sleep(0.1)


def test_runtime_measurement():
    runtime = measure_runtime(
        dummy_function
    )

    assert runtime > 0
