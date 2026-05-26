import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

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
    
    def plot_state_diagram(self, transition_probs):
        # otomata durumlarını ve geçiş olasılıklarını gösteren state diagram çizilir
        if not transition_probs:
            print("HATA | Çizilecek geçiş matrisi bulunamadı.")
            return

        # G : Graph
        G = nx.DiGraph()

        # geçiş olasılıklarına göre yönlü bir grafik oluşturulur
        for current_state, next_states in transition_probs.items():
            for next_state, prob in next_states.items():
                if prob > 0:  
                    G.add_edge(current_state, next_state, weight=prob)

        plt.figure(figsize=(12, 8))
        
        # düğümlerin ekrandaki yerleşimi için spring layout kullanılır
        pos = nx.spring_layout(G, seed=42) 

        # kenar kalınlıkları olasılık ağırlıklarına göre ayarlanır
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]

        # grafik çizilir
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
                node_size=3000, font_size=10, font_weight='bold', width=weights,
                connectionstyle="arc3,rad=0.1")

        # okların üzerine olasılık etiketleri yazılır
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_weight='bold')

        plt.title("Otomata Durum Geçiş Diyagramı (State Diagram)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
