from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

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

def calculate_precision(y_true, y_pred):
    return precision_score(
        y_true,
        y_pred
    )


def calculate_recall(y_true, y_pred):
    return recall_score(
        y_true,
        y_pred
    )
    
def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(
        y_true,
        y_pred
    )
