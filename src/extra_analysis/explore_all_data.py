import pandas as pd

print("--- Анализ DNS (177 МБ) ---")
dns_cols = ['time', 'src', 'resolved']
df_dns = pd.read_csv('data/dns.txt.gz', compression='gzip', header=None, names=dns_cols, nrows=200000)
# признаки: сколько уникальных узлов резолвит хост
dns_features = df_dns.groupby('src').agg(dns_unique_dst=('resolved', 'nunique'), dns_total_queries=('resolved', 'count')).reset_index()
print(dns_features.head())
dns_features.to_csv('data/extra_dns_features.csv', index=False)

print("\n--- Анализ Auth (7.2 ГБ, читаем кусок) ---")
auth_cols = ['time', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type', 'direction', 'success']
# читаем всего 300к строк (авторизаций), чтобы проверить структуру
df_auth = pd.read_csv('data/auth.txt.gz', compression='gzip', header=None, names=auth_cols, nrows=300000)
# фичи по исходному компьютеру
auth_features = df_auth.groupby('src_computer').agg(
    auth_unique_users=('src_user', 'nunique'),
    auth_total_events=('src_user', 'count'),
    auth_failed=('success', lambda x: (x == 'Fail').sum()) # считаем неудачные попытки
).reset_index()
print(auth_features.head())
auth_features.to_csv('data/extra_auth_features.csv', index=False)

print("\n--- Анализ Proc (2.2 ГБ, читаем кусок) ---")
proc_cols = ['time', 'user', 'computer', 'process', 'start_end']
df_proc = pd.read_csv('data/proc.txt.gz', compression='gzip', header=None, names=proc_cols, nrows=200000)
proc_features = df_proc.groupby('computer').agg(
    proc_unique_processes=('process', 'nunique'),
    proc_total_events=('process', 'count')
).reset_index()
print(proc_features.head())
proc_features.to_csv('data/extra_proc_features.csv', index=False)

print("\n--- Анализ RedTeam (4.8 КБ, весь файл) ---")
red_cols = ['time', 'user', 'src_computer', 'dst_computer']
df_red = pd.read_csv('data/redteam.txt.gz', compression='gzip', header=None, names=red_cols)
print(df_red.head())
# помечаем скомпрометированные хосты (1 = скомпрометирован)
red_compromised = df_red[['src_computer']].drop_duplicates()
red_compromised['is_compromised'] = 1
red_compromised.to_csv('data/extra_redteam_labels.csv', index=False)
print("Скрипт завершен. Все файлы сохранены в data/extra_*.csv")