import pandas as pd

# 1 - DNS Features
dns_chunks = []
for chunk in pd.read_csv('data/dns.txt.gz', compression='gzip', header=None, names=['time', 'src', 'resolved'], chunksize=500000):
    dns_chunks.append(chunk.groupby('src').agg(dns_unique_dst=('resolved', 'nunique'), dns_total_queries=('resolved', 'count')))
dns_features = pd.concat(dns_chunks).groupby('src').sum().reset_index()

# 2 - Auth Features (300к строк для чтения, для прода потом весь)
auth_chunks = []
for chunk in pd.read_csv('data/auth.txt.gz', compression='gzip', header=None, names=['time', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type', 'direction', 'success'], chunksize=500000):
    auth_chunks.append(chunk.groupby('src_computer').agg(
        auth_unique_users=('src_user', 'nunique'),
        auth_total_events=('src_user', 'count'),
        auth_failed=('success', lambda x: (x == 'Fail').sum())
    ))
auth_features = pd.concat(auth_chunks).groupby('src_computer').sum().reset_index()

# 3 - Proc Features
proc_chunks = []
for chunk in pd.read_csv('data/proc.txt.gz', compression='gzip', header=None, names=['time', 'user', 'computer', 'process', 'start_end'], chunksize=500000):
    proc_chunks.append(chunk.groupby('computer').agg(
        proc_unique_processes=('process', 'nunique'),
        proc_total_events=('process', 'count')
    ))
proc_features = pd.concat(proc_chunks).groupby('computer').sum().reset_index()

# 4 - RedTeam Features (метка "скомпрометирован")
red_cols = ['time', 'user', 'src_computer', 'dst_computer']
df_red = pd.read_csv('data/redteam.txt.gz', compression='gzip', header=None, names=red_cols)
red_compromised = df_red[['src_computer']].drop_duplicates()
red_compromised['is_compromised'] = 1
red_compromised.to_csv('data/extra_redteam_labels.csv', index=False)

# объединяем в один файл
final_features = dns_features.rename(columns={'src': 'host'})
final_features = final_features.merge(auth_features.rename(columns={'src_computer': 'host'}), on='host', how='outer')
final_features = final_features.merge(proc_features.rename(columns={'computer': 'host'}), on='host', how='outer')
final_features = final_features.merge(red_compromised, on='host', how='left')
final_features['is_compromised'] = final_features['is_compromised'].fillna(0).astype(int)

# пропуски - 0
final_features = final_features.fillna(0)

print("Итоговые дополнительные фичи:")
print(final_features.head())
print(f"Всего хостов: {len(final_features)}")

final_features.to_csv('data/full_extra_features.csv', index=False)
print("Сохранено в data/full_extra_features.csv")