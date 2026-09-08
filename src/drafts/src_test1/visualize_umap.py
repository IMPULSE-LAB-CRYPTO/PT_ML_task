import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# загружаем фичи
df = pd.read_csv('data/host_features.csv')

# берем только числовые столбцы для обучения UMAP (исключение ID хоста)
X = df.drop('host', axis=1)

# стандартизация
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# обучаем umap (снижаем размерность до 2D для картинки)
reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
embedding = reducer.fit_transform(X_scaled)

# draw
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], s=10)
plt.title("Визуализация хостов (без кластеров, просто UMAP)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig('data/umap_plot.png')
plt.show()

print("График сохранен в data/umap_plot.png")