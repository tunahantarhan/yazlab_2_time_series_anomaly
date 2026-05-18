from src.data_loader import load_dataset
from src.data_inspection import get_dataset_shape


def run_experiment(dataset_path):
    df = load_dataset(dataset_path)

    rows, cols = get_dataset_shape(df)

    return rows, cols
