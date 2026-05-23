import json
from src.automata_core import AutomataPreprocessor
from src.data_loader import load_dataset

class ParameterAnalyzer:
    def __init__(self):
        # otomata ön işleme modülü başlatılır
        self.preprocessor = AutomataPreprocessor()

    def run_grid_search(self, raw_data, sizes=[3, 4, 5, 6]):
        # verilen boyut listesindeki tüm kombinasyonlar denenir
        results = {}
        
        print("\n-> Parametre Analiz Döngüsü (Grid Search) Başlıyor...")
        for w in sizes:
            for a in sizes:
                discrete_data = self.preprocessor.extract_patterns(
                    time_series=raw_data, 
                    subsequence_length=w, 
                    alphabet_size=a
                )

                transition_probs = self.preprocessor.calculate_transition_probabilities(discrete_data)
                state_count = len(list(set(discrete_data)))

                key = f"w:{w}_a:{a}"
                results[key] = {
                    "subsequence_length": w,
                    "alphabet_size": a,
                    "state_count": state_count,
                    "transition_matrix_size": len(transition_probs)
                }
                print(f"   -> Analiz tamamlandı: {key} | Bulunan Durum (State) Sayısı: {state_count}")
                
        return results

if __name__ == "__main__":
    analyzer = ParameterAnalyzer()
    
    print("-> Analiz için gerçek veri bekleniyor...")
    
    # TODO: Veri seti yolları kontrol edilip alttaki satırların yorumları kaldırılacak
    # raw_data = load_dataset("data/BATADAL_train.csv")
    # final_results = analyzer.run_grid_search(raw_data, sizes=[3, 4, 5, 6])
    
    # print("\n ==== TÜM ANALİZ SONUÇLARI ====")
    # print(json.dumps(final_results, indent=4, ensure_ascii=False))