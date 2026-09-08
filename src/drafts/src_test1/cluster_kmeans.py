import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

# щагрузка
df = pd.read_csv('data/host_features.csv')
X = df.drop('host', axis=1)

# нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# обучение K-Means (K=3, n_init=10 - избегаем зависимости от случайного старта)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# визуализация (UMAP)
reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
embedding = reducer.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='viridis', s=10)
plt.title("кластеризация хостов (K-Means, K=3)")
plt.savefig('data/umap_clusters.png')
plt.show()

# метки
df['cluster'] = labels
df.to_csv('data/host_features_clustered.csv', index=False)
print("кластеры добавлены в csv")