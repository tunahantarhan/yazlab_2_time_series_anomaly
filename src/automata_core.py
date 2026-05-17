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
    
    def extract_patterns(self, time_series, subsequence_length=None):
        # zaman serisi üzerinde kayan pencere (sliding window) uygulayarak örüntü (pattern) çıkarılır
        # eğer özel olarak parametre gönderilmediyse merkezi config'den alınır
        if subsequence_length is None:
            subsequence_length = config['automata']['subsequence_length']
            
        patterns = []
        # seri baştan sona birim birim kaydırılarak dolaşılır
        for i in range(len(time_series) - subsequence_length + 1):
            window = time_series[i : i + subsequence_length]
            
            # paa uygulanır
            paa_result = self.apply_paa(window)
            
            # sax uygulanır
            sax_result = self.apply_sax(paa_result)
            
            # çıkan sembolleri tire ile birleştirip state oluşturulur ("0-1-2" vb.)
            pattern_str = "-".join(map(str, sax_result))
            patterns.append(pattern_str)
            
        return patterns

    def calculate_transition_probabilities(self, patterns):
        # çıkarılan örüntüler (pattern/state) arasındaki geçiş olasılıkları frekans tabanlı hesaplanır
        transition_counts = {}
        
        # stateler arası geçişler sayılır
        for i in range(len(patterns) - 1):
            current_state = patterns[i]
            next_state = patterns[i + 1]
            
            if current_state not in transition_counts:
                transition_counts[current_state] = {}
            if next_state not in transition_counts[current_state]:
                transition_counts[current_state][next_state] = 0
                
            transition_counts[current_state][next_state] += 1
            
        # geçiş sayıları olasılıklara (0.0 - 1.0) çevrilir
        probabilities = {}
        for current_state, next_states in transition_counts.items():
            total_transitions = sum(next_states.values())
            probabilities[current_state] = {}
            for next_state, count in next_states.items():
                probabilities[current_state][next_state] = count / total_transitions
                
        return probabilities