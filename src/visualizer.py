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
        
    def plot_anomalies_on_timeseries(self, data, anomaly_indices, title="Zaman Serisi Üzerinde Anomaliler"):
        # zaman serisi verisi çizilir ve tespit edilen anomaliler kırmızı noktalarla işaretlenir
        if not isinstance(data, list) and not hasattr(data, '__iter__'):
            print("Geçerli bir zaman serisi sağlanamadı.")
            return

        plt.figure(figsize=(15, 6))
        
        # zaman serisi çizilir
        plt.plot(data, label="Orijinal Veri", color='royalblue', alpha=0.7, linewidth=1.5)
        
        # anomali indeksleri varsa, bu indekslerdeki değerler kırmızı noktalarla işaretlenir
        if anomaly_indices:
            # sadece anomali indekslerine karşılık gelen değerler alınır
            anomaly_values = [data[i] for i in anomaly_indices]
            plt.scatter(anomaly_indices, anomaly_values, color='red', label="Tespit Edilen Anomali", s=50, zorder=5)
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Zaman Adımı (Time Step)", fontsize=12)
        plt.ylabel("Sensör / Sinyal Değeri", fontsize=12)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()