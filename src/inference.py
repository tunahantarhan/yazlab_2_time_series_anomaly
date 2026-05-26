import time


def measure_inference_time(model, X):
    start = time.time()

    model.predict(X, verbose=0)

    end = time.time()

    return end - start
