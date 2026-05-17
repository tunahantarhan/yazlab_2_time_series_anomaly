import numpy as np
import yaml
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config.yaml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

class AutomataPreprocessor:
    def __init__(self):
        self.window_size = config['automata']['window_size']
        self.alphabet_size = config['automata']['alphabet_size']
        
        # SAX için standart normal dağılım (N(0,1)) kesim noktaları
        # proje yönergesine uygun olarak 3, 4, 5 ve 6 için hard-coded değerler.
        self.breakpoints = {
            3: [-0.43, 0.43],
            4: [-0.67, 0.0, 0.67],
            5: [-0.84, -0.25, 0.25, 0.84],
            6: [-0.97, -0.43, 0.0, 0.43, 0.97]
        }

    def apply_paa(self, time_series):
        # PAA (Piecewise Aggregate Approximation) dönüşümü uygulanır.
        # seriyi window_size adedinde eşit (veya yaklaşıksal) parçaya böleriz
        splits = np.array_split(time_series, self.window_size)
        # her bir parçanın ortalamasını alarak boyutu düşürürüz
        return np.array([np.mean(split) for split in splits])

    def apply_sax(self, paa_data):
        # SAX (Symbolic Aggregate Approximation) dönüşümü uygulanır
        # SAX'ın doğru çalışması için verinin Z-normalize edilmesi (mean=0, std=1) şarttır.
        std_dev = np.std(paa_data)
        if std_dev == 0:
            normalized_data = paa_data - np.mean(paa_data)
        else:
            normalized_data = (paa_data - np.mean(paa_data)) / std_dev
            
        # alfabeye uygun kesim noktaları seçilir
        bp = self.breakpoints[self.alphabet_size]
        
        # np.digitize verinin hangi aralığa düştüğünü sembolik indis (0, 1, 2...) olarak döndürür
        sax_symbols = np.digitize(normalized_data, bp)
        
        return sax_symbols