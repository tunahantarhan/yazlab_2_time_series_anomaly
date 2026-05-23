import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hybrid_detector import HybridAnomalyDetector

def test_calculate_hybrid_score():
    # alpha = 0.5 -> dl_score ve automata_score eşit (%50) ağırlıkta 
    detector = HybridAnomalyDetector(alpha = 0.5)
    
    dl_score = 0.8  
    automata_path_prob = 0.2  
    
    # beklenen skor -> (0.5 * 0.8) + (0.5 * (1 - 0.2)) = 0.4 + 0.4 = 0.8
    hybrid_score = detector.calculate_hybrid_score(dl_score, automata_path_prob)
    assert hybrid_score == 0.8