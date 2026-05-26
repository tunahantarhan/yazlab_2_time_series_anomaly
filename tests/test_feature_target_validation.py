import pandas as pd
import pytest

from src.feature_target_split import split_features_and_target


def test_split_features_raises_error_when_target_missing():
    df = pd.DataFrame({
        "sensor_1": [1.0, 2.0, 3.0],
        "sensor_2": [4.0, 5.0, 6.0]
    })

    with pytest.raises(ValueError):
        split_features_and_target(
            df,
            target_column="label"
        )
