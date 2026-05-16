from src.data_loader import load_dataset
from src.data_inspection import get_dataset_shape


def test_dataset_shape():
    df = load_dataset(
        "data/swat/sample.csv"
    )

    rows, cols = get_dataset_shape(df)

    assert rows == 3
    assert cols == 3
