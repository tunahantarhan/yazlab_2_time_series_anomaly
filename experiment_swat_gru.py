import pandas as pd

from src.preprocessing_pipeline import prepare_dataset
from src.sequence_generator import create_labeled_sequences
from src.models.gru_model import build_gru_model
from src.trainer import train_model


print("SWAT yükleniyor...")

df = pd.read_csv(
    "data/swat/merged.csv"
)

df["Normal/Attack"] = (
    df["Normal/Attack"]
    .map({
        "Normal": 0,
        "Attack": 1
    })
)

dataset = prepare_dataset(
    df,
    target_column="Normal/Attack",
    columns_to_drop=["Timestamp"]
)

X_train_seq, y_train_seq = create_labeled_sequences(
    dataset["X_train"],
    dataset["y_train"],
    window_size=5
)

X_val_seq, y_val_seq = create_labeled_sequences(
    dataset["X_val"],
    dataset["y_val"],
    window_size=5
)

model = build_gru_model(
    input_shape=(
        X_train_seq.shape[1],
        X_train_seq.shape[2]
    )
)

history = train_model(
    model,
    X_train_seq,
    y_train_seq,
    X_val_seq,
    y_val_seq
)

print("Eğitim tamamlandı.")
