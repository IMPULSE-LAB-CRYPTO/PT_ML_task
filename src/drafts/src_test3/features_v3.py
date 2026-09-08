import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']

# берем те же 1 млн строк (не перегружаю память)
chunk_size = 500000
frames = []
for chunk in pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=cols, chunksize=chunk_size):
    frames.append(chunk)
    if len(frames) * chunk_size > 1000000:
        break
df = pd.concat(frames)

# 1 - собираем вектор для каждого хоста.
# берем и src_port, и dst_port
# превращаем в строку вида "443 443 80 8080 53"
def create_port_doc(group):
    ports = list(group['dst_port'].astype(str)) + list(group['src_port'].astype(str))
    return " ".join(ports)

print("Собираем текстовые профили по портам...")
port_texts = df.groupby('src').apply(create_port_doc, include_groups=False).reset_index()
port_texts.columns = ['host', 'port_text']

# 2 - считаем TF-IDF
vectorizer = TfidfVectorizer(max_features=2000) # Берем топ 2000 портов
tfidf_matrix = vectorizer.fit_transform(port_texts['port_text'])

# 3 - превращаем в датафрейм
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"port_{name}" for name in feature_names])
tfidf_df['host'] = port_texts['host'].values

print(f"Создано TF-IDF признаков: {tfidf_df.shape[1] - 1} для {len(tfidf_df)} хостов.")
tfidf_df.to_csv('data/port_tfidf_features.csv', index=False)
print("Сохранено в data/port_tfidf_features.csv")