import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AutomataVisualizer:
    def plot_transition_heatmap(self, transition_probs):
        # durum geçiş matrisi ısı haritası olarak görselleştirilir
        if not transition_probs:
            print("HATA | Görselleştirilecek geçiş matrisi bulunamadı.")
            return

        # iç içe sözlük yapısı pandas dataframe'e çevrilir
        df = pd.DataFrame(transition_probs).fillna(0.0) # eksik değerler 0.0 olarak doldurulur
        # satır ve sütunları düzgün bir formatta göstermek için matris transpoze edilir
        df = df.T 

        plt.figure(figsize=(10, 8))
        # seaborn heatmap ile açıklanabilirlik sağlanır
        sns.heatmap(df, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=.5)
        
        plt.title("Otomata Durum Geçiş Olasılıkları (Transition Probabilities)")
        plt.xlabel("Hedef Durum (Next State)")
        plt.ylabel("Mevcut Durum (Current State)")
        plt.tight_layout()
        plt.show()