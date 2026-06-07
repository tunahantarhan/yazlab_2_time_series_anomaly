# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.automata_core import AutomataPreprocessor


WINDOW_SIZE = 4
ALPHABET_SIZE = 3
THRESHOLD = 0.001


print("Automata metrik deneyi baslatiliyor...")

df = pd.read_csv(
    "data/swat/merged.csv"
)

df.columns = df.columns.str.strip()
df = df.ffill().bfill()

y = (
    df["Normal/Attack"]
    .map({
        "Normal": 0,
        "Attack": 1
    })
    .values
)

if "Timestamp" in df.columns:
    df = df.drop(
        columns=["Timestamp"]
    )

if "Normal/Attack" in df.columns:
    df = df.drop(
        columns=["Normal/Attack"]
    )

sensor_column = df.columns[0]

print(f"Secilen sensor sutunu: {sensor_column}")

series = df[sensor_column].head(50000).values
labels = y[:50000]

train_end = int(
    len(series) * 0.60
)

val_end = int(
    len(series) * 0.80
)

train_series = series[:train_end]
test_series = series[val_end:]

test_labels = labels[val_end:]

preprocessor = AutomataPreprocessor()

train_patterns = preprocessor.extract_patterns(
    time_series=train_series,
    subsequence_length=WINDOW_SIZE,
    alphabet_size=ALPHABET_SIZE
)

test_patterns = preprocessor.extract_patterns(
    time_series=test_series,
    subsequence_length=WINDOW_SIZE,
    alphabet_size=ALPHABET_SIZE
)

transition_probs = preprocessor.calculate_transition_probabilities(
    train_patterns
)

known_patterns = list(
    set(train_patterns)
)

predictions = []

for i in range(len(test_patterns)):

    pattern = test_patterns[i]

    if pattern not in known_patterns:
        predictions.append(1)
        continue

    if i == 0:
        predictions.append(0)
        continue

    previous_pattern = test_patterns[i - 1]

    if previous_pattern not in known_patterns:
        previous_pattern = preprocessor.find_closest_pattern(
            previous_pattern,
            known_patterns
        )

    transition_probability = (
        transition_probs
        .get(previous_pattern, {})
        .get(pattern, 0)
    )

    if transition_probability < THRESHOLD:
        predictions.append(1)
    else:
        predictions.append(0)

min_len = min(
    len(test_labels),
    len(predictions)
)

test_labels_aligned = test_labels[:min_len]
predictions = predictions[:min_len]


predictions = np.array(predictions)

accuracy = accuracy_score(
    test_labels_aligned,
    predictions
)

precision = precision_score(
    test_labels_aligned,
    predictions,
    zero_division=0
)

recall = recall_score(
    test_labels_aligned,
    predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels_aligned,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    test_labels_aligned,
    predictions
)

print()
print("=== AUTOMATA METRICS RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print()
print("Confusion Matrix")
print(cm)
