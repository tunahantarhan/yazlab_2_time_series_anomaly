import numpy as np
import os
import sys

# fonksiyonları import etmek adına src klasörü path'e eklenir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automata_core import AutomataPreprocessor

def test_apply_paa():
    preprocessor = AutomataPreprocessor()
    
    # test için config kuralı ezilerek hard-coded değer atanır
    preprocessor.window_size = 3 
    
    # 6 elemanlı seri 3 parçaya bölünür: [1,3], [5,7], [9,11]
    time_series = np.array([1, 3, 5, 7, 9, 11])
    expected_paa = np.array([2.0, 6.0, 10.0])
    
    paa_result = preprocessor.apply_paa(time_series)
    np.testing.assert_almost_equal(paa_result, expected_paa)

def test_apply_sax():
    preprocessor = AutomataPreprocessor()
    preprocessor.alphabet_size = 4
    
    # veri normalize edilecek ve 4 sembole (0, 1, 2, 3) ayrılacak
    # z-normalize olunca kabaca: [-1.34, -0.34, 0.28, 1.40] değerlerini alacak
    # 4'lük alfabede kesim noktaları: -0.67, 0.0, 0.67
    paa_data = np.array([-1.0, -0.2, 0.3, 1.2])
    expected_sax = np.array([0, 1, 2, 3]) 
    
    sax_result = preprocessor.apply_sax(paa_data)
    np.testing.assert_array_equal(sax_result, expected_sax)