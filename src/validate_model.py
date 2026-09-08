import sys
import pandas as pd
sys.path.append('src')
from main import store

# 10 хостов из датасета для проверки
test_hosts = store.df['host'].head(10).tolist()

print("ПРОВЕРКА МОДЕЛИ НА 10 ХОСТАХ")

for host in test_hosts:
    try:
        similar = store.get_similar(host, top_n=3)
        print(f"\n🔍 Хост: {host}")
        for res in similar:
            print(f"   -> {res['host']} (Sim: {res['similarity']}) | Порты: {res['common_ports']} | Поведение: {res['behavior_traits']}")
    except Exception as e:
        print(f"Ошибка для {host}: {e}")
        
print("\nПроверка завершена")