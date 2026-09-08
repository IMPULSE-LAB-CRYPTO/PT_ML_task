import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

df_behavior = pd.read_csv('data/host_features_v2.csv')
df_ports = pd.read_csv('data/port_tfidf_features.csv')

df = pd.merge(df_behavior, df_ports, on='host', how='inner')

# создаем 3 варианта признаков
X_behavior = df_behavior.drop('host', axis=1)
X_ports = df_ports.drop('host', axis=1)
X_combined = df.drop('host', axis=1)

# масштабируем
sc_b = StandardScaler().fit_transform(X_behavior)
sc_p = StandardScaler().fit_transform(X_ports)
sc_c = StandardScaler().fit_transform(X_combined)

for name, X in [("Only Behavioral", sc_b), ("Only Ports (TF-IDF)", sc_p), ("Combined", sc_c)]:
    knn = NearestNeighbors(n_neighbors=2, metric='cosine')
    knn.fit(X)
    # ищем ближайшего соседа для C10
    idx = df.index[df['host'] == 'C10'][0]
    distances, indices = knn.kneighbors(X[idx].reshape(1, -1), n_neighbors=2)
    print(f"--- {name} ---")
    print(f"Top similar host: {df.iloc[indices[0][1]]['host']}, Similarity: {1 - distances[0][1]:.2f}")