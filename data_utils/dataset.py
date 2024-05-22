import os
import pandas as pd

def load_data(config):
    path = os.path.join(config.data.path, config.data.name + '.csv')
    data = pd.read_csv(path)
    data = data.iloc[:, 2:6]
    data['combined'] = data.apply(lambda row: f'COURTCASE: {row["text1"]} COURTHOLDING: {row["text2"]}', axis=1)
    data = data[data['label'] == 1]
    return data
