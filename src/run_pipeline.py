import subprocess
import sys

# сколько строк читаем по умолчанию (быстро дл теста)
SAMPLE_ROWS = 500000 

print("Запуск пайплайна (быстрый режим, 500к строк)...")
subprocess.run([sys.executable, "src/features_v2.py"], check=True)
subprocess.run([sys.executable, "src/features_v3.py"], check=True)
subprocess.run([sys.executable, "src/extra_analysis/build_extra_features.py"], check=True)
subprocess.run([sys.executable, "src/extra_analysis/build_investigation_features.py"], check=True)
subprocess.run([sys.executable, "src/make_graph.py"], check=True)
subprocess.run([sys.executable, "src/make_ui_data.py"], check=True)
print("Пайплайн успешно отработал. Далее запускаем API и UI")