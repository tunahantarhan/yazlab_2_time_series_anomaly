import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analyzer import ParameterAnalyzer

# parametre analiz döngüsü ve kombinasyon üretimi test edilir
def test_run_grid_search():
    analyzer = ParameterAnalyzer()
    
    test_sizes = [3, 4]
    
    # mock raw veriler ile analiz çalıştırılır
    mock_raw_data = [10, 12, 14, 11, 15, 50, 12, 13]
    results = analyzer.run_grid_search(mock_raw_data, test_sizes)
    
    # farklı 2 window_size ve 2 alphabet_size ile toplam 4 kombinasyon
    assert len(results) == 4
    
    # sonuç içeriği kontrol edilir
    assert "w:3_a:4" in results
    assert "state_count" in results["w:3_a:4"]