import pandas as pd
import numpy as np
import os
import sys

# fonksiyonları import etmek adına src klasörü path'e eklenir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import split_time_series, scale_data_no_leakage

def test_split_time_series():
    # 100 satırlık sahte time series verisi
    df = pd.DataFrame({'value': np.arange(100)})
    train, val, test = split_time_series(df)
    
    # oranların 60/20/20 olduğu doğrulanır
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    
    # zaman serisinin kronolojik bölündüğü doğrulanır (shuffle=False)
    assert train.iloc[-1]['value'] == 59
    assert val.iloc[0]['value'] == 60
    assert val.iloc[-1]['value'] == 79
    assert test.iloc[0]['value'] == 80

def test_scale_data_no_leakage():
    # mock train, val, test veri setleri
    train = pd.DataFrame({'sensor_1': [10, 20, 30], 'sensor_2': [100, 200, 300]})
    val = pd.DataFrame({'sensor_1': [40, 50], 'sensor_2': [400, 500]})
    test = pd.DataFrame({'sensor_1': [60, 70], 'sensor_2': [600, 700]})
    
    train_scaled, val_scaled, test_scaled, scaler = scale_data_no_leakage(
        train, val, test, columns_to_scale=['sensor_1', 'sensor_2']
    )
    
    # scaler'ın sadece train'e fit edildiği doğrulanır (data leakage kontrolü)
    # train setinin ortalaması 0 ve standart sapması 1 olmalıdır
    np.testing.assert_almost_equal(train_scaled['sensor_1'].mean(), 0.0)
    np.testing.assert_almost_equal(train_scaled['sensor_1'].std(ddof=0), 1.0)
    
    # leakage kontrolü adına validation ve test'in ortalaması 0 olmamalıdır 
    assert val_scaled['sensor_1'].mean() != 0.0
    assert test_scaled['sensor_1'].mean() != 0.0