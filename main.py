import os
import pandas as pd

from src.automata_core import AutomataPreprocessor
from src.hybrid_detector import HybridAnomalyDetector
from src.visualizer import AutomataVisualizer
from src.data_loader import load_dataset


def main():
    print("-> Hibrit Anomali Tespit Sistemi Başlatılıyor...")

    preprocessor = AutomataPreprocessor()
    detector = HybridAnomalyDetector(alpha=0.5)
    visualizer = AutomataVisualizer()

    print("\n-> Eğitim verisi yükleniyor ve automata matrisleri oluşturuluyor...")

    # eğitim için batadal verisi seçildi, veri seti  uyuşmazlığı düzeltildi
    train_data_raw = load_dataset("data/batadal/training_dataset_1.csv")

    train_data_raw.columns = train_data_raw.columns.str.strip()
    train_data_raw = train_data_raw.ffill().bfill()

    if "DATETIME" in train_data_raw.columns:
        train_data_raw = train_data_raw.drop(columns=["DATETIME"])
    if "ATT_FLAG" in train_data_raw.columns:
        train_data_raw = train_data_raw.drop(columns=["ATT_FLAG"])

    # ilk sensör sütunu (1D) ayrıştırılarak automata matrisi oluşturulur
    sensor_column = train_data_raw.columns[0]
    print(f"-> Analiz için seçilen sensör sütunu: {sensor_column}")

    train_series = train_data_raw[sensor_column].head(10000).values
    train_data_discrete = preprocessor.extract_patterns(train_series)
    known_patterns = list(set(train_data_discrete))

    transition_probs = preprocessor.calculate_transition_probabilities(
        train_data_discrete
    )

    print(
        f"-> Durum Geçiş Matrisi hesaplandı. Toplam state: {len(known_patterns)}"
    )

    print("\n-> Test verisi işleniyor...")

    test_data_raw = load_dataset("data/batadal/test_dataset.csv")
    test_data_raw.columns = test_data_raw.columns.str.strip()
    test_data_raw = test_data_raw.ffill().bfill()

    if "DATETIME" in test_data_raw.columns:
        test_data_raw = test_data_raw.drop(columns=["DATETIME"])

    # test verisinde de aynı sensör sütunu işlenir
    test_series = test_data_raw[sensor_column].head(1000).values
    test_sequence = preprocessor.extract_patterns(test_series)

    recovered_sequence = []
    for pattern in test_sequence:
        if pattern not in known_patterns:
            closest = preprocessor.find_closest_pattern(
                pattern, known_patterns
            )
            recovered_sequence.append(closest)
        else:
            recovered_sequence.append(pattern)

    path_prob = preprocessor.calculate_path_probability(
        recovered_sequence, transition_probs
    )

    print(f"-> Otomata Yol Olasılığı Path Probability: {path_prob:.4f}")

    dl_score = 1.0 - path_prob
    print(f"-> DL skoru geçici olarak hesaplandı: {dl_score:.4f}")

    hybrid_score = detector.calculate_hybrid_score(dl_score, path_prob)
    print(f"FINAL HİBRİT ANOMALİ SKORU: {hybrid_score:.4f}")

    if hybrid_score > 0.70:
        print("SİSTEM UYARISI: Zaman serisinde anomali tespit edildi!")
    else:
        print("BAŞARILI: Sistem normal seyrinde devam ediyor.")

    print("\n-> Açıklanabilirlik heatmap'i çiziliyor...")
    visualizer.plot_transition_heatmap(transition_probs)


if __name__ == "__main__":
    main()