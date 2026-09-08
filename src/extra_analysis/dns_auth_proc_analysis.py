import pandas as pd

# читаем 100к строк dns
print("Анализируем DNS...")
dns_cols = ['time', 'src', 'resolved']
df_dns = pd.read_csv('data/dns.txt.gz', compression='gzip', header=None, names=dns_cols, nrows=100000)
print(df_dns.head())

# считаем количество уникальных резолвов на хост
dns_counts = df_dns.groupby('src').agg(unique_dns=('resolved', 'nunique')).reset_index()
print(dns_counts.head())

# читаем 100к строк auth
print("Анализируем Auth...")
auth_cols = ['time', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type', 'direction', 'success']
df_auth = pd.read_csv('data/auth.txt.gz', compression='gzip', header=None, names=auth_cols, nrows=100000)
# ччитаем уникальных пользователей на хост
auth_counts = df_auth.groupby('src_computer').agg(unique_users=('src_user', 'nunique')).reset_index()
print(auth_counts.head())

# сохраняем таблицы для дальнейшего объединения
dns_counts.to_csv('data/extra_dns_features.csv', index=False)
auth_counts.to_csv('data/extra_auth_features.csv', index=False)
print("Скрипт завершен. Сохранено в data/extra_*.csv")