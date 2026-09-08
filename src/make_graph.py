import pandas as pd
import pickle

cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']

print("Строим детальный граф (с байтами и портами)...")
graph = {} # {src: {dst: {'bytes':..., 'ports':...}}}
chunk_size = 500000
for chunk in pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=cols, chunksize=chunk_size):
    for _, row in chunk.iterrows():
        src, dst = row['src'], row['dst']
        if src not in graph: graph[src] = {}
        if dst not in graph[src]: graph[src][dst] = {'bytes': 0, 'ports': set()}
        graph[src][dst]['bytes'] += row['bytes']
        graph[src][dst]['ports'].add(row['dst_port'])
    if sum(len(v) for v in graph.values()) > 10000000: break

with open('data/network_graph.pkl', 'wb') as f:
    pickle.dump(graph, f)
print(f"Граф с данными сохранен. Узлов: {len(graph)}")