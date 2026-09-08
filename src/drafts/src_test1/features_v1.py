import pandas as pd


cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']
# читаем все данные по чанкам
chunk_size = 500000
frames = []
for chunk in pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=cols, chunksize=chunk_size):
    # для теста возьмем только первые 500k строк, быстро проверяем признаки
    frames.append(chunk)
    if len(frames) * chunk_size > 1000000: # ограничим пока 1 млн строк
        break

df = pd.concat(frames)

# Группировка 1: Агрегация по Источнику (Source Computer)
# для каждого источника считаем, сколько он отправлял трафика, сколько принимал, и тд
# для этого выясняем: сколько всего байт отправил, сколько пакетов, уникальных портов и тд
features = df.groupby('src').agg({
    'dst': 'nunique',                  # уникальные получатели
    'src_port': 'nunique',             # сколько уникальных портов использовал
    'dst_port': 'nunique',             # сколько портов назначения (серверов) трогал
    'bytes': 'sum',                    # сколько всего байт передал
    'packets': 'sum',                  # сколько всего пакетов отправил
    'duration': 'mean'                 # средняя длительность сессии
}).reset_index()

# переименуем колонки для читаемости
features.columns = ['host', 'unique_dst', 'unique_src_port', 'unique_dst_port', 'total_bytes', 'total_packets', 'avg_duration']

# добавляем соотношение отправленных байт к количеству пакетов (средний размер пакета)
features['avg_packet_size'] = features['total_bytes'] / features['total_packets'].replace(0, 1)

# логируем: сколько хостов мы получили
print(f"Успешно создано признаков для {len(features)} хостов")
print(features.head())

# сохраняем в csv, для визуализации
features.to_csv('data/host_features.csv', index=False)