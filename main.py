import os
import json
import pandas as pd
import yaml

from src.automata_core import AutomataPreprocessor
from src.hybrid_detector import HybridAnomalyDetector
from src.visualizer import AutomataVisualizer
from src.data_loader import load_dataset

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    print("-> Hibrit Anomali Tespit Sistemi Başlatılıyor...")

    preprocessor = AutomataPreprocessor()
    # alpha değerini config'den alacak şekilde veya varsayılan 0.5 olarak ayarlıyoruz
    detector = HybridAnomalyDetector(alpha=config['model']['hybrid_alpha'])
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

    print(f"-> Durum Geçiş Matrisi hesaplandı. Toplam state: {len(known_patterns)}")

    print("\n-> Test verisi işleniyor ve JSON Raporu hazırlanıyor...")

    test_data_raw = load_dataset("data/batadal/test_dataset.csv")
    test_data_raw.columns = test_data_raw.columns.str.strip()
    test_data_raw = test_data_raw.ffill().bfill()

    if "DATETIME" in test_data_raw.columns:
        test_data_raw = test_data_raw.drop(columns=["DATETIME"])

    # test verisinde de aynı sensör sütunu işlenir
    test_series = test_data_raw[sensor_column].head(1000).values
    test_sequence = preprocessor.extract_patterns(test_series)

    decision_logs = []

    # JSON loglama işlemi
    for i in range(len(test_sequence) - 1):
        current_pattern = test_sequence[i]
        next_pattern = test_sequence[i+1]
        
        status = "seen"
        mapped_to = current_pattern
        
        # eğer pattern daha önce eğitimde görülmemişse levenshtein ile en yakını bulunur
        if current_pattern not in known_patterns:
            status = "unseen"
            mapped_to = preprocessor.find_closest_pattern(current_pattern, known_patterns)
            
        # olasılık hesaplaması
        prob = 0.0
        if mapped_to in transition_probs and next_pattern in transition_probs.get(mapped_to, {}):
            prob = transition_probs[mapped_to][next_pattern]
            
        # geçici dl skoru ve hibrit skor hesabı
        dl_score = 1.0 - prob
        hybrid_score = detector.calculate_hybrid_score(dl_score, prob)
        
        decision = "anomaly" if hybrid_score > 0.70 else "normal"
        
        # istenen formatta JSON objesi oluşturuluyor
        log_entry = {
            "state": str(mapped_to),
            "pattern": str(current_pattern),
            "status": status,
            "mapped_to": str(mapped_to),
            "probability": round(float(prob), 4),
            "decision": decision,
            "confidence": round(float(hybrid_score), 4)
        }
        decision_logs.append(log_entry)

    # logların json dosyasına kaydedilmesi
    os.makedirs("output", exist_ok=True)
    json_path = "output/confidence_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(decision_logs, f, indent=4, ensure_ascii=False)

    print(f"-> BAŞARILI: {json_path} dosyası oluşturuldu ve skorlar kaydedildi.")
    
    print("\n-> Açıklanabilirlik heatmap'i çiziliyor...")
    visualizer.plot_transition_heatmap(transition_probs)

if __name__ == "__main__":
    main()