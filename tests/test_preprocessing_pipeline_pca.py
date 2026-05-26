import pandas as pd

from src.preprocessing_pipeline import prepare_dataset


def test_prepare_dataset_applies_pca():
    df = pd.DataFrame({
        "sensor_1": [1, 2, 3, 4, 5],
        "sensor_2": [5, 4, 3, 2, 1],
        "label": [0, 0, 1, 0, 1]
    })

    result = prepare_dataset(
        df,
        target_column="label"
    )

    assert result["X_train"].shape[1] == 1
