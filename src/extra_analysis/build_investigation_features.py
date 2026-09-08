import pandas as pd
import numpy as np

print("Загружаем данные для расследования...")

# 1 - Анализ времени - берем первые 500к строк flows
flows_cols = ['time', 'duration', 'src', 'src_port', 'dst', 'dst_port', 'protocol', 'packets', 'bytes']
chunks = []
for chunk in pd.read_csv('data/flows.txt.gz', compression='gzip', header=None, names=flows_cols, chunksize=500000):
    chunks.append(chunk)
    if len(chunks) * 500000 > 1000000: break
df_flows = pd.concat(chunks)
time_shift = df_flows.groupby('src').agg(
    min_time=('time', 'min'),
    max_time=('time', 'max'),
    total_bytes=('bytes', 'sum')
)
# метрика: активность во второй половине времени относительно первой
median_time = time_shift['min_time'].median() + (time_shift['max_time'].median() - time_shift['min_time'].median()) / 2
time_shift['time_shift_ratio'] = time_shift['total_bytes'] / (time_shift['min_time'].median() + 1)

# 2 - DNS Tunneling (Подозрительный DNS)
dns_features = pd.read_csv('data/full_extra_features.csv')[['host', 'dns_total_queries', 'dns_unique_dst']]
# если 100+ запросов и 20+ уникальных резолвов
dns_features['dns_suspicious'] = ((dns_features['dns_total_queries'] > 100) & (dns_features['dns_unique_dst'] > 20)).astype(int)

# 3 - Брутфорс - берем 500к строк auth
auth_chunks = []
for chunk in pd.read_csv('data/auth.txt.gz', compression='gzip', header=None, names=['time', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type', 'direction', 'success'], chunksize=500000):
    auth_chunks.append(chunk)
    if len(auth_chunks) * 500000 > 1000000: break
df_auth = pd.concat(auth_chunks)
auth_metrics = df_auth.groupby('src_computer').agg(
    auth_failed=('success', lambda x: (x == 'Fail').sum()),
    auth_total_events=('success', 'count')
).reset_index().rename(columns={'src_computer': 'host'})
auth_metrics['brute_force_suspicious'] = ((auth_metrics['auth_failed'] > 50) & (auth_metrics['auth_failed'] / auth_metrics['auth_total_events'] > 0.3)).astype(int)

# 4 - Редкие процессы - берем первые 500к строк proc
proc_chunks = []
for chunk in pd.read_csv('data/proc.txt.gz', compression='gzip', header=None, names=['time', 'user', 'computer', 'process', 'start_end'], chunksize=500000):
    proc_chunks.append(chunk)
    if len(proc_chunks) * 500000 > 1000000: break
df_proc = pd.concat(proc_chunks)

# получаем список зараженных хостов
red_cols = ['time', 'user', 'src_computer', 'dst_computer']
df_red = pd.read_csv('data/redteam.txt.gz', compression='gzip', header=None, names=red_cols)
compromised_hosts = df_red['src_computer'].unique()

# находим процессы, которые запускаются ТОЛЬКО на зараженных хостах (IOC)
all_proc = df_proc.groupby('computer')['process'].apply(set).to_dict()
common_processes = set()
for h in compromised_hosts:
    if h in all_proc:
        common_processes.update(all_proc[h])
        
# считаем, где встречаются эти процессы
proc_on_normal = set()
for h, procs in all_proc.items():
    if h not in compromised_hosts:
        proc_on_normal.update(procs)

rare_procs = common_processes - proc_on_normal
print(f"Найдено редких (вредоносных) процессов: {len(rare_procs)}")

# объединяем все
final_features = dns_features.merge(auth_metrics, on='host', how='outer')
final_features = final_features.merge(time_shift[['time_shift_ratio']], left_on='host', right_index=True, how='left')

# добавляем метку редких процессов (если у хоста есть подозрительные процессы)
def has_rare_process(row):
    if row['host'] in all_proc:
        if rare_procs.intersection(all_proc[row['host']]):
            return 1
    return 0

final_features['rare_process_suspicious'] = final_features.apply(has_rare_process, axis=1)
final_features = final_features.fillna(0)

# save
final_features.to_csv('data/investigation_features.csv', index=False)
print("Готово! Сохранено в data/investigation_features.csv")