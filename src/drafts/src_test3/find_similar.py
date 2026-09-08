import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# загружаем оба файла с признаками
df_behavior = pd.read_csv('data/host_features_v2.csv')
df_ports = pd.read_csv('data/port_tfidf_features.csv')

# объединяем их по колонке 'host' (внутреннее соединение, тк хосты совпадают)
df = pd.merge(df_behavior, df_ports, on='host', how='inner')

# убираем ID хоста для обучения
X = df.drop('host', axis=1)

# стандартизация (для UMAP и KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# обучение модели поиска соседей (KNN)
# metric='cosine' - косинусная близость, которая игнорирует масштаб векторов
# metric='euclidean' - тоже ок, но косинус чаще используется для текстовых/поведенческих фич
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(X_scaled)

# Функция поиска похожих
def find_similar_hosts(query_host, top_n=5):
    if query_host not in df['host'].values:
        return f"Хост {query_host} не найден в обученной модели."
    
    # получаем индекс запрашиваемого хоста
    idx = df.index[df['host'] == query_host][0]
    
    # ищем ближайших соседей
    distances, indices = knn.kneighbors(X_scaled[idx].reshape(1, -1), n_neighbors=top_n + 1)
    
    results = []
    for i in range(1, top_n + 1): # Пропуск самого себя (индекс 0)
        host_name = df.iloc[indices[0][i]]['host']
        similarity = 1 - distances[0][i] # косинусное расстояние -> близость (1 - dist)
        results.append((host_name, round(similarity, 2)))
        
    return results

# ТЕСТ движка
# берем любого хоста из датасета
test_host = df.iloc[0]['host']
print(f"Ищем похожих для: {test_host}")
neighbors = find_similar_hosts(test_host, top_n=5)

for host, score in neighbors:
    print(f"  {host} -> {score}")

# показываем, какие порты использовал сам запрашиваемый хост и его сосед
print("\n ОБЪЯСНЕНИЕ (какие порты общие)")
for host, score in neighbors:
    # Ищем общие порты
    q_ports = set(df[df['host'] == test_host].filter(regex='port_').iloc[0][lambda x: x > 0].index)
    n_ports = set(df[df['host'] == host].filter(regex='port_').iloc[0][lambda x: x > 0].index)
    common_ports = list(q_ports.intersection(n_ports))[:5] # Первые 5 общих
    print(f"С {host}: общие порты -> {common_ports}")