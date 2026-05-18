from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(
        y_true,
        y_pred
    )

def calculate_f1_score(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred
    )
