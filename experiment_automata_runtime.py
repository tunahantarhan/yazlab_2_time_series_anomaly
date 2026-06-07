# -*- coding: utf-8 -*-

import time
import pandas as pd

from src.automata_core import AutomataPreprocessor


print("Automata runtime deneyi baslatiliyor...")

df = pd.read_csv(
    "data/swat/merged.csv"
)

df.columns = df.columns.str.strip()
df = df.ffill().bfill()

if "Timestamp" in df.columns:
    df = df.drop(
        columns=["Timestamp"]
    )

if "Normal/Attack" in df.columns:
    df = df.drop(
        columns=["Normal/Attack"]
    )

numeric_df = df.select_dtypes(
    include=["number"]
)

sensor_column = numeric_df.columns[0]

print(f"Secilen sensor sutunu: {sensor_column}")

series = numeric_df[sensor_column].head(50000).values

preprocessor = AutomataPreprocessor()

train_start = time.time()

patterns = preprocessor.extract_patterns(
    time_series=series,
    subsequence_length=5,
    alphabet_size=6
)

transition_probs = preprocessor.calculate_transition_probabilities(
    patterns
)

training_time = time.time() - train_start

inference_start = time.time()

for pattern in patterns[:1000]:
    if pattern not in transition_probs:
        _ = preprocessor.find_closest_pattern(
            pattern,
            list(transition_probs.keys())
        )

inference_time = time.time() - inference_start

print()
print("=== AUTOMATA RUNTIME RESULTS ===")
print(f"Training Time : {training_time:.4f} sn")
print(f"Inference Time: {inference_time:.4f} sn")
