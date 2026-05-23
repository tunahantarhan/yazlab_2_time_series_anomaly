import os
import pandas as pd
from src.automata_core import AutomataPreprocessor
from src.hybrid_detector import HybridAnomalyDetector
from src.visualizer import AutomataVisualizer
from src.data_loader import load_dataset
from src.models.gru_model import build_gru_model

def main():
    print("-> Hibrit Anomali Tespit Sistemi Başlatılıyor...")
    
    # modüller başlatılır
    preprocessor = AutomataPreprocessor()
    detector = HybridAnomalyDetector(alpha=0.5)
    visualizer = AutomataVisualizer()
    
    # ==== VERİ YÜKLEME VE EĞİTİM (TRAIN) ====
    print("\n-> Eğitim verisi yükleniyor ve automata matrisleri oluşturuluyor...")
    
    # TODO: Dataloader modülünü kullanarak gerçek BATADAL/SWAT train verisi çekilecek
    # train_data_raw = load_dataset("data/BATADAL_train.csv")
    train_data_discrete = preprocessor.extract_patterns(train_data_raw)
    
    # modelin bildiği tüm durumlar (sözlük)
    known_patterns = list(set(train_data_discrete))
    
    # durum geçiş olasılıkları hesaplanır 
    transition_probs = preprocessor.calculate_transition_probabilities(train_data_discrete)
    print(f"-> Durum Geçiş Matrisi BAŞARIYLA hesaplandı! Toplam durum (state): {len(known_patterns)}")

    # ==== TEST VE ANOMALİ TESPİTİ ====
    print("\n-> Test verisi işleniyor...")
    
    # TODO: Test verisi buraya yüklenecek
    # test_data_raw = load_dataset("data/BATADAL_test.csv")
    test_sequence = preprocessor.extract_patterns(test_data_raw) # "0-1-1" anomali
    
    # levenshtein ile bilinmeyen pattern'ların en yakın komşusunu bulma
    recovered_sequence = []
    for pattern in test_sequence:
        if pattern not in known_patterns:
            print(f"-> DİKKAT: Daha önce görülmemiş örüntü yakalandı ['{pattern}']")
            closest = preprocessor.find_closest_pattern(pattern, known_patterns)
            print(f"    -> Levenshtein ile kurtarıldı: ['{closest}']")
            recovered_sequence.append(closest)
        else:
            recovered_sequence.append(pattern)
            
    # automata yol olasılığı hesaplanır
    path_prob = preprocessor.calculate_path_probability(recovered_sequence, transition_probs)
    print(f"-> Otomata Yol Olasılığı (Path Probability): {path_prob:.4f}")
    
    # derin öğrenme modelinin yeniden yapılandırma hatası (reconstruction error) hesaplanır
    # TODO: Eğitilen modelin test verisindeki Reconstruction Error'u buraya eklenecek
    #dl_score = my_gru_model.predict(test_data_raw)
    
    print(f"-> DL Yeniden Yapılandırma Hatası (Reconstruction Error): {dl_score:.4f}")
    
    # final karar mekanizması
    hybrid_score = detector.calculate_hybrid_score(dl_score, path_prob)
    print(f"FİNAL HİBRİT ANOMALİ SKORU: {hybrid_score:.4f}")
    
    if hybrid_score > 0.70:
        print("SİSTEM UYARISI: Zaman serisinde anomali tespit edildi!")
    else:
        print("BAŞARILI: Sistem normal seyrinde devam ediyor.")
        
    # ==== GÖRSELLEŞTİRME (AÇIKLANABİLİRLİK) ====
    print("\n-> Açıklanabilirlik heatmap'i çiziliyor...")
    visualizer.plot_transition_heatmap(transition_probs)
    
    # TODO: eğer orijinal raw data varsa zaman serisi de çizilebilir
    # visualizer.plot_anomalies_on_timeseries(raw_data_array, anomaly_indices)

if __name__ == "__main__":
    main()