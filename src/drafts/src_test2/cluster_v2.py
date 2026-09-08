import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import umap

# загрузка
df = pd.read_csv('data/host_features_v2.csv')
X = df.drop('host', axis=1)

# стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# находим аномалии с помощью Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42) # 5% данных считаем аномальными
outlier_labels = iso_forest.fit_predict(X_scaled)

# добавляем метку аномалии к датафрейму
df['outlier'] = outlier_labels

# разделяем данные
normal_data = X_scaled[outlier_labels == 1]
outlier_data = X_scaled[outlier_labels == -1]

print(f"Обычных хостов: {len(normal_data)}, Аномальных: {len(outlier_data)}")

# Обучаем K-Means только на нормальных данных
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(normal_data)

# восстанавливаю метки для всех (аномалии получают кластер -1)
full_labels = np.full(len(df), -1)
full_labels[outlier_labels == 1] = labels
df['cluster'] = full_labels

# Визуализация UMAP
reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
embedding = reducer.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
# аномалии в красный цвет
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df['cluster'], palette='viridis', s=10, legend='full')
plt.title("Визуализация v2 (Изоляция выбросов + Кластеризация)")
plt.legend(title='Cluster (-1 = Аномалия)')
plt.savefig('data/umap_clusters_v2.png')
plt.show()

df.to_csv('data/host_features_clustered_v2.csv', index=False)
print("Готово, сохранено в data/host_features_clustered_v2.csv")