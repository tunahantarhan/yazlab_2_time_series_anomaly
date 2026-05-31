import numpy as np


def create_sequences(data, window_size):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y


def create_labeled_sequences(X, y, window_size):
    X_sequences = []
    y_sequences = []

    for i in range(len(X) - window_size):
        X_sequences.append(
            X[i:i + window_size]
        )

        y_sequences.append(
            y[i + window_size]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )
