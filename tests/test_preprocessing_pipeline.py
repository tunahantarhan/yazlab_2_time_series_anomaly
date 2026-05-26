import pandas as pd

from src.preprocessing_pipeline import prepare_dataset


def test_prepare_dataset_pipeline():
    df = pd.DataFrame({
        "timestamp": ["t1", "t2", "t3", "t4", "t5"],
        "sensor_1": [1, 2, 3, 4, 5],
        "sensor_2": [5, 4, 3, 2, 1],
        "label": [0, 0, 1, 0, 1]
    })

    result = prepare_dataset(
        df,
        target_column="label",
        columns_to_drop=["timestamp"]
    )

    assert "X_train" in result
    assert "X_val" in result
    assert "X_test" in result
    assert "y_train" in result
    assert "y_val" in result
    assert "y_test" in result
