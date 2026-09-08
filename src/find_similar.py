import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# загружаем данные
df_behavior = pd.read_csv('data/host_features_v2.csv')
df_ports = pd.read_csv('data/port_tfidf_features.csv')

# объединяем
df = pd.merge(df_behavior, df_ports, on='host', how='inner')
X = df.drop('host', axis=1)

# нормализуем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# обучаем KNN
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(X_scaled)

def get_common_ports(host1, host2, top_n=3):
    # получаем индексы признаков, где значение > 0
    ports1 = set(df[df['host'] == host1].filter(regex='port_').iloc[0][lambda x: x > 0].index)
    ports2 = set(df[df['host'] == host2].filter(regex='port_').iloc[0][lambda x: x > 0].index)
    common = list(ports1.intersection(ports2))
    
    # превращаем port_445 в 445
    real_ports = [p.replace('port_', '') for p in common][:top_n]
    return real_ports

def find_similar_hosts(query_host, top_n=5):
    if query_host not in df['host'].values:
        return f"Хост {query_host} не найден."
    
    idx = df.index[df['host'] == query_host][0]
    distances, indices = knn.kneighbors(X_scaled[idx].reshape(1, -1), n_neighbors=top_n + 1)
    
    results = []
    print(f"\nИщем похожих для: {query_host}")
    for i in range(1, top_n + 1):
        host_name = df.iloc[indices[0][i]]['host']
        similarity = 1 - distances[0][i] # 1 - расстояние = близость
        common_ports = get_common_ports(query_host, host_name)
        results.append((host_name, similarity, common_ports))
        
        # выводим в консоль для проверки
        print(f"  {host_name} -> {similarity:.2f} | Общие порты: {common_ports}")
    return results

# тест
test_host = df.iloc[0]['host'] # Берем первый хост
neighbors = find_similar_hosts(test_host, top_n=5)