import pandas as pd

from src.feature_target_split import split_features_and_target


def test_split_features_and_target():
    df = pd.DataFrame({
        "sensor_1": [1.0, 2.0, 3.0],
        "sensor_2": [4.0, 5.0, 6.0],
        "label": [0, 1, 0]
    })

    X, y = split_features_and_target(
        df,
        target_column="label"
    )

    assert "label" not in X.columns
    assert len(y) == 3
    assert X.shape == (3, 2)
