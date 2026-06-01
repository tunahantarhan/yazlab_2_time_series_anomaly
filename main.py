import os
import pandas as pd

from src.automata_core import AutomataPreprocessor
from src.hybrid_detector import HybridAnomalyDetector
from src.visualizer import AutomataVisualizer
from src.data_loader import load_dataset


def main():
    print("-> Hibrit Anomali Tespit Sistemi Baslatiliyor...")

    preprocessor = AutomataPreprocessor()
    detector = HybridAnomalyDetector(alpha=0.5)
    visualizer = AutomataVisualizer()

    print("\n-> Egitim verisi yukleniyor ve automata matrisleri olusturuluyor...")

    train_data_raw = load_dataset(
        "data/swat/merged.csv"
    )

    train_data_raw.columns = train_data_raw.columns.str.strip()
    train_data_raw = train_data_raw.ffill()
    train_data_raw = train_data_raw.bfill()

    if "Timestamp" in train_data_raw.columns:
        train_data_raw = train_data_raw.drop(
            columns=["Timestamp"]
        )

    if "Normal/Attack" in train_data_raw.columns:
        train_data_raw["Normal/Attack"] = (
            train_data_raw["Normal/Attack"]
            .map({
                "Normal": 0,
                "Attack": 1
            })
        )

    train_data_raw = train_data_raw.head(50000)

    train_data_discrete = preprocessor.extract_patterns(
        train_data_raw
    )

    known_patterns = list(set(train_data_discrete))

    transition_probs = preprocessor.calculate_transition_probabilities(
        train_data_discrete
    )

    print(
        f"-> Durum Gecis Matrisi hesaplandi! Toplam state: {len(known_patterns)}"
    )

    print("\n-> Test verisi isleniyor...")

    test_data_raw = load_dataset(
        "data/batadal/test_dataset.csv"
    )

    test_data_raw.columns = test_data_raw.columns.str.strip()
    test_data_raw = test_data_raw.ffill()
    test_data_raw = test_data_raw.bfill()

    if "DATETIME" in test_data_raw.columns:
        test_data_raw = test_data_raw.drop(
            columns=["DATETIME"]
        )

    test_data_raw = test_data_raw.head(5000)

    test_sequence = preprocessor.extract_patterns(
        test_data_raw
    )

    recovered_sequence = []

    for pattern in test_sequence:
        if pattern not in known_patterns:
            print(f"-> Dikkat: Unseen pattern yakalandi: {pattern}")
            closest = preprocessor.find_closest_pattern(
                pattern,
                known_patterns
            )
            print(f"    -> Levenshtein ile eslendi: {closest}")
            recovered_sequence.append(closest)
        else:
            recovered_sequence.append(pattern)

    path_prob = preprocessor.calculate_path_probability(
        recovered_sequence,
        transition_probs
    )

    print(
        f"-> Otomata Yol Olasiligi Path Probability: {path_prob:.4f}"
    )

    dl_score = 1.0 - path_prob

    print(
        f"-> DL skoru gecici olarak hesaplandi: {dl_score:.4f}"
    )

    hybrid_score = detector.calculate_hybrid_score(
        dl_score,
        path_prob
    )

    print(
        f"FINAL HIBRIT ANOMALI SKORU: {hybrid_score:.4f}"
    )

    if hybrid_score > 0.70:
        print("SISTEM UYARISI: Zaman serisinde anomali tespit edildi!")
    else:
        print("BASARILI: Sistem normal seyrinde devam ediyor.")

    print("\n-> Aciklanabilirlik heatmap'i ciziliyor...")
    visualizer.plot_transition_heatmap(
        transition_probs
    )


if __name__ == "__main__":
    main()
