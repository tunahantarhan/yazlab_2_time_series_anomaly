from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense


def build_gru_model(input_shape):
    model = Sequential([
        GRU(32, input_shape=input_shape),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
