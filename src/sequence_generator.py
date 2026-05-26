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
