from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense


def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            input_shape=input_shape
        ),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
