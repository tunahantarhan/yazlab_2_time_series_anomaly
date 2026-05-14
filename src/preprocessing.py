import pandas as pd
import yaml
import os
from sklearn.preprocessing import StandardScaler

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config.yaml')

# config.yaml okuma
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# zaman serisi bozulmadan %60 train, %20 val, %20 test olacak şekilde bölme
def split_time_series(df):
    n = len(df)
    train_end = int(n * config['data']['train_split'])
    val_end = train_end + int(n * config['data']['val_split'])
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    
    return train_data, val_data, test_data

# data leakage engellemek adına sadece train seti üzerinden fit yapma
def scale_data_no_leakage(train, val, test, columns_to_scale):
    scaler = StandardScaler()
    
    # sadece train üzerinden fit
    train_scaled = train.copy()
    train_scaled[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])
    
    # val ve test üzerinden sadece transform
    val_scaled = val.copy()
    val_scaled[columns_to_scale] = scaler.transform(val[columns_to_scale])
    
    test_scaled = test.copy()
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    return train_scaled, val_scaled, test_scaled, scaler