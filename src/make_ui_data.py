import pandas as pd
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# 1 - загружаем базовые фичи
df_behavior = pd.read_csv('data/host_features_v2.csv')
df_ports = pd.read_csv('data/port_tfidf_features.csv')

# 2 - загружаем доп. фичи (DNS, Auth, Proc) и убираем оттуда is_compromised, дабы не перезаписать
df_extra = pd.read_csv('data/full_extra_features.csv')
df_extra = df_extra.drop(columns=['is_compromised'])

# объединяем
df = pd.merge(df_behavior, df_ports, on='host', how='inner')
df = pd.merge(df, df_extra, on='host', how='left')
df = df.fillna(0)

# 3 - читаем redteam
red_cols = ['time', 'user', 'src_computer', 'dst_computer']
df_red = pd.read_csv('data/redteam.txt.gz', compression='gzip', header=None, names=red_cols)

# логика - источник атаки (src_computer) = скомпрометированный хост
compromised_sources = df_red[['src_computer']].drop_duplicates().rename(columns={'src_computer': 'host'})
compromised_sources['is_compromised'] = 1
print(f"Найдено скомпрометированных хостов в RedTeam: {len(compromised_sources)}")
print(f"Это хосты: {compromised_sources['host'].tolist()}")

# 4 - добавляем недостающие зараженные хосты в таблицу (создаем нул векторы)
missing_red = compromised_sources[~compromised_sources['host'].isin(df['host'])]
if not missing_red.empty:
    print(f"Добавляем их на карту: {missing_red['host'].tolist()}")
    dummy_df = pd.DataFrame(0, index=range(len(missing_red)), columns=df.columns)
    dummy_df['host'] = missing_red['host'].values
    dummy_df['is_compromised'] = 1
    df = pd.concat([df, dummy_df], ignore_index=True)
else:
    print("Все зараженные хосты уже есть в данных flows.")

# 5 - подготовка признаков
behavior_cols = ['total_bytes', 'avg_bytes', 'total_packets', 'avg_packets', 'avg_duration', 'entropy_dst', 'avg_packet_size', 'ratio_bytes_per_packet']
scaled_behavior = StandardScaler().fit_transform(df[behavior_cols])

port_cols = [col for col in df.columns if col.startswith('port_')]
svd = TruncatedSVD(n_components=50, random_state=42)
svd_ports = svd.fit_transform(df[port_cols])

extra_cols = ['dns_unique_dst', 'dns_total_queries', 'auth_unique_users', 'auth_total_events', 'auth_failed', 'proc_unique_processes', 'proc_total_events']
scaled_extra = StandardScaler().fit_transform(df[extra_cols])

X_combined = np.hstack((scaled_behavior, svd_ports, scaled_extra))

# 6 - кластеризация и umap
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_combined)

# логика - находим кластеры, где меньше 20 объектов - аномалии (-1)
cluster_counts = pd.Series(clusters).value_counts()
small_clusters = cluster_counts[cluster_counts < 20].index.tolist()

if small_clusters:
    print(f"Обнаружены слишком маленькие кластеры (выбросы): {small_clusters}. Переводим их в статус Аномалии (-1).")
    clusters = np.where(np.isin(clusters, small_clusters), -1, clusters)

reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
embedding = reducer.fit_transform(X_combined)

# 7 - сохраняем в ui_umap.csv
df_ui = df[['host', 'is_compromised']].copy()
df_ui['x'] = embedding[:, 0]
df_ui['y'] = embedding[:, 1]
df_ui['cluster'] = clusters

# проверяем итог
print(f"Итог: Всего хостов на карте: {len(df_ui)}. Из них скомпрометированных: {df_ui['is_compromised'].sum()}")

df_ui.to_csv('data/ui_umap.csv', index=False)
print("Готово. Сохранено в data/ui_umap.csv")