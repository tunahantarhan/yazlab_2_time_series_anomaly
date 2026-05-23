import json
import os

class ScoreLogger:
    def __init__(self, filepath="data/confidence_scores.json"):
        # json dosyasının kaydedileceği yol ve log listesi tanımlanır
        self.filepath = filepath
        self.logs = []
        
        # log dosyasının kaydedileceği dizin yoksa oluşturulur
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def log_decision(self, step, dl_score, automata_prob, hybrid_score, is_anomaly):
        # her bir karar adımı sözlük formatında listeye eklenir
        log_entry = {
            "step": step,
            "dl_reconstruction_error": round(float(dl_score), 4),
            "automata_path_probability": round(float(automata_prob), 4),
            "hybrid_confidence_score": round(float(hybrid_score), 4),
            "is_anomaly": is_anomaly
        }
        self.logs.append(log_entry)
        self.save_to_json()

    def save_to_json(self):
        # log listesi json formatında dosyaya yazılır
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)