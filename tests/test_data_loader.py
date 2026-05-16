import pandas as pd

from src.data_loader import load_dataset


def test_dataset_loading():
    df = load_dataset(
        "data/swat/sample.csv"
    )

    assert isinstance(df, pd.DataFrame)
