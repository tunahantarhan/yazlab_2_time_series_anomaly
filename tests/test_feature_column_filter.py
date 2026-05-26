import pandas as pd

from src.feature_filter import drop_non_feature_columns


def test_drop_non_feature_columns():
    df = pd.DataFrame({
        "timestamp": ["t1", "t2", "t3"],
        "sensor_1": [1.0, 2.0, 3.0],
        "sensor_2": [4.0, 5.0, 6.0],
        "source_file": ["a.csv", "a.csv", "b.csv"],
        "label": [0, 1, 0]
    })

    filtered_df = drop_non_feature_columns(
        df,
        columns_to_drop=["timestamp", "source_file"]
    )

    assert "timestamp" not in filtered_df.columns
    assert "source_file" not in filtered_df.columns
    assert "sensor_1" in filtered_df.columns
    assert "label" in filtered_df.columns
