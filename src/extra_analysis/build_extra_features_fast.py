import pandas as pd

print("1. Читаем DNS (1 млн строк)...")
dns_df = pd.read_csv('data/dns.txt.gz', compression='gzip', header=None, 
                     names=['time', 'src', 'resolved'], nrows=1000000)
dns_features = dns_df.groupby('src').agg(
    dns_unique_dst=('resolved', 'nunique'),
    dns_total_queries=('resolved', 'count')
).reset_index().rename(columns={'src': 'host'})
print(f"Готово: {len(dns_features)} хостов\n")

print("2. Читаем Auth (1 млн строк)...")
auth_df = pd.read_csv('data/auth.txt.gz', compression='gzip', header=None, 
                      names=['time', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 
                             'auth_type', 'logon_type', 'direction', 'success'], nrows=1000000)
auth_features = auth_df.groupby('src_computer').agg(
    auth_unique_users=('src_user', 'nunique'),
    auth_total_events=('src_user', 'count'),
    auth_failed=('success', lambda x: (x == 'Fail').sum())
).reset_index().rename(columns={'src_computer': 'host'})
print(f"Готово: {len(auth_features)} хостов\n")

print("3. Читаем Proc (500 тыс строк)...")
proc_df = pd.read_csv('data/proc.txt.gz', compression='gzip', header=None, 
                      names=['time', 'user', 'computer', 'process', 'start_end'], nrows=500000)
proc_features = proc_df.groupby('computer').agg(
    proc_unique_processes=('process', 'nunique'),
    proc_total_events=('process', 'count')
).reset_index().rename(columns={'computer': 'host'})
print(f"Готово: {len(proc_features)} хостов\n")

print("4. Читаем RedTeam (весь файл)..")
red_cols = ['time', 'user', 'src_computer', 'dst_computer']
df_red = pd.read_csv('data/redteam.txt.gz', compression='gzip', header=None, names=red_cols)
red_compromised = df_red[['src_computer']].drop_duplicates().rename(columns={'src_computer': 'host'})
red_compromised['is_compromised'] = 1
print(f"Готово: {len(red_compromised)} скомпрометированных хостов\n")

print("5. Объединяем все фичи...")
final_features = dns_features.merge(auth_features, on='host', how='outer')
final_features = final_features.merge(proc_features, on='host', how='outer')
final_features = final_features.merge(red_compromised, on='host', how='left')

# заполняем пропуски и меняем None на 0
final_features['is_compromised'] = final_features['is_compromised'].fillna(0).astype(int)
final_features = final_features.fillna(0)

print(f"Итого уникальных хостов: {len(final_features)}")
final_features.to_csv('data/full_extra_features.csv', index=False)
print("Успешно сохранено в data/full_extra_features.csv")