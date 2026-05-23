import sys
import os
import pytest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualizer import AutomataVisualizer

@patch("src.visualizer.plt.show") # grafik ekranının test sırasında açılıp testi bloklaması engellenir
def test_plot_transition_heatmap(mock_show):
    visualizer = AutomataVisualizer()
    
    # mock transition olasılıkları
    transition_probs = {
        "0-1": {"1-2": 0.5, "2-0": 0.5},
        "1-2": {"2-0": 1.0}
    }
    
    visualizer.plot_transition_heatmap(transition_probs)
    # "plt.show()" fonksiyonunun en az bir kere çağrıldığı doğrulanır
    mock_show.assert_called_once()
    
@patch("src.visualizer.plt.show")
def test_plot_anomalies_on_timeseries(mock_show):
    visualizer = AutomataVisualizer()
    
    # mock zaman serisi
    data = [10, 12, 11, 15, 50, 12, 13, 14, 55, 11]
    
    # mock anomali indeksleri
    anomaly_indices = [4, 8]
    
    visualizer.plot_anomalies_on_timeseries(data, anomaly_indices)
    mock_show.assert_called_once()