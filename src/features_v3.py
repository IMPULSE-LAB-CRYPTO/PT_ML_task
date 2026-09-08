import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']

chunk_size = 500000
frames = []
for chunk in pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=cols, chunksize=chunk_size):
    frames.append(chunk)
    if len(frames) * chunk_size > 1000000:
        break
df = pd.concat(frames)

# исправление: исключаем эфемерные 'N' порты
def create_port_doc(group):
    # отфильтровываем мусорные порты
    dst_ports = group['dst_port'].astype(str)
    dst_ports = dst_ports[~dst_ports.str.startswith('N')] 
    
    src_ports = group['src_port'].astype(str)
    src_ports = src_ports[~src_ports.str.startswith('N')]
    
    # склеиваем
    return " ".join(list(dst_ports) + list(src_ports))

print("Собираем текстовые профили по портам (только чистые порты)...")
port_texts = df.groupby('src').apply(create_port_doc, include_groups=False).reset_index()
port_texts.columns = ['host', 'port_text']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=500) # Сократили до 500 самых важных портов (убрать шум)
tfidf_matrix = vectorizer.fit_transform(port_texts['port_text'])

feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"port_{name}" for name in feature_names])
tfidf_df['host'] = port_texts['host'].values

print(f"Создано TF-IDF признаков: {tfidf_df.shape[1] - 1} для {len(tfidf_df)} хостов.")
tfidf_df.to_csv('data/port_tfidf_features.csv', index=False)
print("Сохранено в data/port_tfidf_features.csv")