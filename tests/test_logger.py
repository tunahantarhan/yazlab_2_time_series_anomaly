import os
import json
import pytest
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import ScoreLogger

def test_log_confidence_score(tmp_path):
    # pytest'in geçici (tmp) klasöründe bir test dosyası yolu oluşturulur
    log_file = tmp_path / "test_scores.json"
    logger = ScoreLogger(filepath=str(log_file))
    
    # örnek bir karar loglanır
    logger.log_decision(
        step="0-1-1", 
        dl_score=0.85, 
        automata_prob=0.10, 
        hybrid_score=0.80, 
        is_anomaly=True
    )
    
    # dosya gerçekten oluşturuldu mu diye kontrol edilir
    assert os.path.exists(log_file)
    
    # içindeki veriler json olarak okunur ve doğru mu kontrol edilir
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["step"] == "0-1-1"
        assert data[0]["is_anomaly"] == True
        assert data[0]["hybrid_confidence_score"] == 0.80