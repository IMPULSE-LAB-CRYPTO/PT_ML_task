import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# загружаем все фичи (те же как в make_ui_data.py)
df_behavior = pd.read_csv('data/host_features_v2.csv')
df_ports = pd.read_csv('data/port_tfidf_features.csv')
df_extra = pd.read_csv('data/full_extra_features.csv')
df_extra = df_extra.drop(columns=['is_compromised'])

df = pd.merge(df_behavior, df_ports, on='host', how='inner')
df = pd.merge(df, df_extra, on='host', how='left')
df = df.fillna(0)

# масштабируем (как в основном пайплайне)
behavior_cols = ['total_bytes', 'avg_bytes', 'total_packets', 'avg_packets', 'avg_duration', 'entropy_dst', 'avg_packet_size', 'ratio_bytes_per_packet']
scaled_behavior = StandardScaler().fit_transform(df[behavior_cols])

port_cols = [col for col in df.columns if col.startswith('port_')]
svd = TruncatedSVD(n_components=50, random_state=42)
svd_ports = svd.fit_transform(df[port_cols])

extra_cols = ['dns_unique_dst', 'dns_total_queries', 'auth_unique_users', 'auth_total_events', 'auth_failed', 'proc_unique_processes', 'proc_total_events']
scaled_extra = StandardScaler().fit_transform(df[extra_cols])

X_combined = np.hstack((scaled_behavior, svd_ports, scaled_extra))

# считаем WCSS (Within-Cluster Sum of Squares) для K от 1 до 10
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_combined)
    wcss.append(kmeans.inertia_)

# график
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method (Выбор оптимального K)')
plt.xlabel('Число кластеров (K)')
plt.ylabel('WCSS (Сумма квадратов ошибок)')
plt.savefig('data/elbow_plot.png')
print("График сохранен в data/elbow_plot.png")
print("Необходима точка, где резкое падение графика замедляется (Обычно это K=3 или K=4.)")