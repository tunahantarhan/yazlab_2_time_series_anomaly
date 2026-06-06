# -*- coding: utf-8 -*-

import pandas as pd

from src.automata_core import AutomataPreprocessor


TRAIN_RATIO = 0.60
TEST_RATIO = 0.20

WINDOW_SIZE = 5
ALPHABET_SIZE = 6


print("Automata unseen analizi baslatiliyor...")

df = pd.read_csv(
    "data/swat/merged.csv"
)

df.columns = df.columns.str.strip()
df = df.ffill()
df = df.bfill()

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

train_end = int(
    len(series) * TRAIN_RATIO
)

test_start = int(
    len(series) * (1 - TEST_RATIO)
)

train_series = series[:train_end]
test_series = series[test_start:]

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

known_patterns = set(train_patterns)

total_test_patterns = len(test_patterns)
unseen_count = 0
mapped_count = 0

for pattern in test_patterns:
    if pattern not in known_patterns:
        unseen_count += 1

        closest_pattern = preprocessor.find_closest_pattern(
            pattern,
            list(known_patterns)
        )

        if closest_pattern in known_patterns:
            mapped_count += 1

detection_rate = unseen_count / total_test_patterns

if unseen_count == 0:
    mapping_accuracy = 0
else:
    mapping_accuracy = mapped_count / unseen_count

print()
print("=== AUTOMATA UNSEEN ANALYSIS RESULTS ===")
print(f"Total Test Patterns : {total_test_patterns}")
print(f"Unseen Patterns     : {unseen_count}")
print(f"Mapped Patterns     : {mapped_count}")
print(f"Detection Rate      : {detection_rate:.4f}")
print(f"Mapping Accuracy    : {mapping_accuracy:.4f}")
