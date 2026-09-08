import pandas as pd
import numpy as np
from scipy.stats import entropy

cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']

chunk_size = 500000
frames = []
# читаем 1 млн строк, чтобы сохранить скорость, но получить хорошую выборку
for chunk in pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=cols, chunksize=chunk_size):
    frames.append(chunk)
    if len(frames) * chunk_size > 1000000:
        break
df = pd.concat(frames)

# исправленная агрегация: уникальные имена всем колонкам
features = df.groupby('src').agg(
    unique_dst=('dst', 'nunique'),
    unique_src_port=('src_port', 'nunique'),
    unique_dst_port=('dst_port', 'nunique'),
    total_bytes=('bytes', 'sum'),
    avg_bytes=('bytes', 'mean'),
    total_packets=('packets', 'sum'),
    avg_packets=('packets', 'mean'),
    avg_duration=('duration', 'mean'),
    dst_list=('dst', list)
).reset_index()

features.columns = ['host', 'unique_dst', 'unique_src_port', 'unique_dst_port', 
                    'total_bytes', 'avg_bytes', 'total_packets', 'avg_packets', 
                    'avg_duration', 'dst_list']

# ролевые признаки (поведение)
# средний размер пакета
features['avg_packet_size'] = features['total_bytes'] / features['total_packets'].replace(0, 1)

# энтропия распределения по IP назначения (1.0 - общается со всеми равномерно, 0.0 - ходит к 1-2 хостам)
def calc_entropy(ip_list):
    if not ip_list: return 0
    _, counts = np.unique(ip_list, return_counts=True)
    return entropy(counts, base=2) 

features['entropy_dst'] = features['dst_list'].apply(calc_entropy)
features.drop(columns=['dst_list'], inplace=True)

# соотношение отправленных байт к принятым (признак сервера)
features['ratio_bytes_per_packet'] = features['total_bytes'] / features['total_packets'].replace(0, 1)

# ЛОГАРИФМИЧЕСКАЯ ТРАНСФОРМАЦИЯ (!!!)
# сжимаем тяжелые хвосты (убираем влияние аномальных 236 миллионов байт)
for col in ['total_bytes', 'avg_bytes', 'total_packets', 'avg_packets', 'avg_duration', 'avg_packet_size']:
    features[col] = np.log1p(features[col]) # log1p = log(x + 1)

print(f"Создано {len(features)} хостов.")
print(features.head())

features.to_csv('data/host_features_v2.csv', index=False)
print("Сохранено в data/host_features_v2.csv")