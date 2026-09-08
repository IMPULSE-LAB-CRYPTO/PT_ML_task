import pandas as pd

# колонки
cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']

# проверяем работу на первых 100к строках
df = pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=cols, nrows=100000)
print(df.head())
print(df.describe())

# проверка, какие хосты чаще всего встречаются (первые 10)
print(df['src'].value_counts().head(10))