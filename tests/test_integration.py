import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automata_core import AutomataPreprocessor
from src.hybrid_detector import HybridAnomalyDetector

# otomata ve hibrit dedektörün uçtan uca beraber çalışabildiğini test eden fonksiyon
def test_end_to_end_pipeline():
    
    # modüller başlatılır
    preprocessor = AutomataPreprocessor()
    detector = HybridAnomalyDetector(alpha=0.6) # DL'e %60 ağırlık verelim
    
    # mock train verisi ve geçiş olasılıkları
    known_patterns = ["0-1-2", "1-2-1", "2-1-0"]
    transition_probs = {
        "0-1-2": {"1-2-1": 0.8, "2-1-0": 0.2},
        "1-2-1": {"2-1-0": 1.0}
    }
    
    # mock test verisi (anomalili bir durum)
    test_sequence = ["0-1-1", "1-2-1"] # "0-1-1" anomali
    
    # levenshtein mesafesi kullanılarak bilinmeyen patern en yakın bilinen paterne dönüştürülür
    recovered_sequence = []
    for pattern in test_sequence:
        if pattern not in known_patterns:
            closest = preprocessor.find_closest_pattern(pattern, known_patterns)
            recovered_sequence.append(closest)
        else:
            recovered_sequence.append(pattern)
            
    # "0-1-1" -> "0-1-2"
    assert recovered_sequence == ["0-1-2", "1-2-1"]
    
    # path probability hesaplanır
    # P(0-1-2 -> 1-2-1) = 0.8 
    path_prob = preprocessor.calculate_path_probability(recovered_sequence, transition_probs)
    assert path_prob == 0.8
    
    # mock DL anomali skoru 
    mock_dl_score = 0.90
    
    # hibrit skor hesaplanır
    # (0.6 * DL_skoru) + (0.4 * (1 - Path_Prob))
    # (0.6 * 0.90) + (0.4 * (1 - 0.8)) = 0.54 + 0.08 = 0.62
    final_anomaly_score = detector.calculate_hybrid_score(mock_dl_score, path_prob)
    
    # sistemin baştan sona sağlıklı çalıştığı doğrulanır
    assert round(final_anomaly_score, 2) == 0.62