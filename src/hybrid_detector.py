class HybridAnomalyDetector:
    def __init__(self, alpha=0.5):
        # alpha -> derin öğrenme (DL) skorunun ağırlığı (0.0 - 1.0)
        self.alpha = alpha

    def calculate_hybrid_score(self, dl_error, automata_path_prob):
        # DL hatası ve automata yol olasılığını birleştirerek anomali skoru üretir
        automata_anomaly_score = 1.0 - automata_path_prob
        hybrid_score = (self.alpha * dl_error) + ((1.0 - self.alpha) * automata_anomaly_score)
        return hybrid_score