import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# читаем flows по кусочкам (1 млн строк)
cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']
chunks = []
for chunk in pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=cols, chunksize=500000):
    chunks.append(chunk)
    if len(chunks) * 500000 > 1000000:
        break
df_flows = pd.concat(chunks)

# 1 - находим среднюю временную метку (50% всех событий)
mid_time = df_flows['time'].median()
print(f"Середина временного отрезка (t={mid_time})")

# 2 - разбиваем данные на два независимых временных окна
df_window_1 = df_flows[df_flows['time'] < mid_time] # ранний период
df_window_2 = df_flows[df_flows['time'] >= mid_time] # поздний период

# 3 - считаем одинаковые фичи для обоих окон (агрегация по источнику)
def get_features(df):
    return df.groupby('src').agg(
        total_bytes=('bytes', 'sum'),
        total_packets=('packets', 'sum'),
        unique_dst=('dst', 'nunique'),
        avg_duration=('duration', 'mean')
    ).reset_index().rename(columns={'src': 'host'})

features_w1 = get_features(df_window_1)
features_w2 = get_features(df_window_2)

# 4 - оставляем хосты, которые были в обоих окнах
# оцениваем, как меняется поведение одного и того же компьютера
features_common = pd.merge(features_w1, features_w2, on='host', suffixes=('_w1', '_w2'))

# 5 - готовим матрицы X1 (Window 1) и X2 (Window 2)
# используем только колонки, оканчивающиеся на _w1 и _w2
cols_w1 = [c for c in features_common.columns if c.endswith('_w1')]
cols_w2 = [c for c in features_common.columns if c.endswith('_w2')]

X1 = features_common[cols_w1].values
X2 = features_common[cols_w2].values

# 6 - нормализация без утечки данных!
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
# применяем тот же scaler к X2, а не учим новый 
# (важно, чтобы не смотреть в будущее)
X2_scaled = scaler.transform(X2)

# 7 - обучаем K-Means на раннем периоде (Window 1)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_1 = kmeans.fit_predict(X1_scaled)

# 8 - предсказываем кластеры на позднем периоде (Window 2)
# используем ту же самую обученную модель
labels_2 = kmeans.predict(X2_scaled)

# 9 - оцениваем устойчивость кластеров
# Adjusted Rand Index (ARI): 1.0 - идеально, 0.0 - случайно, <0 - хуже случайного
ari_score = adjusted_rand_score(labels_1, labels_2)
nmi_score = normalized_mutual_info_score(labels_1, labels_2)

print(f"\n--- РЕЗУЛЬТАТЫ УСТОЙЧИВОСТИ ВО ВРЕМЕНИ ---")
print(f"Хостов, наблюдаемых в обоих окнах: {len(features_common)}")
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

if ari_score > 0.75:
    print("\nВывод: Кластеры ОЧЕНЬ УСТОЙЧИВЫ. Группы компьютеров практически не меняют своего поведения во времени.")
elif ari_score > 0.5:
    print("\nВывод: Кластеры УМЕРЕННО УСТОЙЧИВЫ. Есть небольшой дрейф, но общая структура сохраняется.")
else:
    print("\nВывод: Кластеры НЕУСТОЙЧИВЫ. Поведение компьютеров сильно меняется со временем, что требует динамических моделей.")