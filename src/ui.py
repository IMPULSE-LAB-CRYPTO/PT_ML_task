import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Кэширование тяжелых файлов
@st.cache_data
def load_ui_data():
    return pd.read_csv('data/ui_umap.csv')

@st.cache_data
def load_invest_data():
    if os.path.exists('data/investigation_features.csv'):
        return pd.read_csv('data/investigation_features.csv')
    return pd.DataFrame()

@st.cache_resource
def load_graph():
    with open('data/network_graph.pkl', 'rb') as f:
        return pickle.load(f)

# Загружаем данные
df_ui = load_ui_data()
df_invest = load_invest_data()
graph = load_graph()

# Объединяем признаки расследования с картой
df_ui = df_ui.merge(df_invest, on='host', how='left').fillna(0)

st.title("Network Traffic Similarity Analyzer")

# Инициализация session_state
if "host_input" not in st.session_state: st.session_state.host_input = "C10"
if "neighbors" not in st.session_state: st.session_state.neighbors = None
if "threat_result" not in st.session_state: st.session_state.threat_result = None

host_input = st.text_input("Введите ID хоста", st.session_state.host_input)
st.session_state.host_input = host_input

# Кнопка поиска похожих
if st.button("Найти похожие хосты"):
    try:
        response = requests.post("http://127.0.0.1:8000/find_similar", json={"host_id": host_input})
        if response.status_code == 200:
            st.session_state.neighbors = response.json()['most_similar']
        else:
            st.error("Хост не найден.")
    except Exception as e:
        st.error(f"Ошибка: {e}")

# Отображение результатов
if st.session_state.neighbors:
    st.subheader("Топ похожих хостов")
    st.dataframe(pd.DataFrame(st.session_state.neighbors))
    
    st.subheader("Карта сети (UMAP)")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_ui, x='x', y='y', hue='cluster', palette='viridis', ax=ax, s=10, alpha=0.6)
    
    # Выделяем аномалии (-1) красным и крупным размером
    anomalies = df_ui[df_ui['cluster'] == -1]
    if not anomalies.empty:
        ax.scatter(anomalies['x'], anomalies['y'], color='red', s=100, edgecolor='black', zorder=6, label='Аномальные узлы')
    
    query_row = df_ui[df_ui['host'] == host_input]
    if not query_row.empty:
        ax.scatter(query_row['x'], query_row['y'], color='red', s=150, edgecolor='white', zorder=10)
        ax.annotate('Искомый', (query_row['x'].values[0], query_row['y'].values[0]), fontsize=10, color='red')
    
    neighbor_ids = [n['host'] for n in st.session_state.neighbors]
    neigh_rows = df_ui[df_ui['host'].isin(neighbor_ids)]
    ax.scatter(neigh_rows['x'], neigh_rows['y'], color='orange', s=100, edgecolor='black', zorder=5, label='Похожие хосты')
    ax.legend(title="Легенда")
    st.pyplot(fig)
    
    st.subheader("Почему они похожи?")
    for n in st.session_state.neighbors:
        explanation = f"**{n['host']}** (близость {n['similarity']})"
        if n['common_ports']: explanation += f": Порты ({', '.join(n['common_ports'])})"
        if n['behavior_traits']: explanation += f" и {', '.join(n['behavior_traits'])}."
        st.write(explanation)

# Панель расследования
with st.expander("Панель аналитика и угроз (Threat Panel)"):
    st.write("### Статистика по кластерам")
    st.dataframe(df_ui.groupby('cluster').size().reset_index(name='count'))
    
    st.write("### Аномальные узлы")
    outliers = df_ui[df_ui['cluster'] == -1]
    if len(outliers) > 0: st.dataframe(outliers.head(10))
    else: st.info("Аномалий не обнаружено.")
    
    st.divider()
    
    st.subheader("Панель расследования угроз")
    compromised_hosts = df_ui[df_ui['is_compromised'] == 1]['host'].tolist()
    
    if not compromised_hosts:
        st.warning("Запусти make_ui_data.py для добавления зараженных хостов.")
    else:
        selected_threat = st.selectbox("Выберите скомпрометированный хост:", compromised_hosts)
        
        if st.button("Построить вектор атаки"):
            try:
                response = requests.post("http://127.0.0.1:8000/threat_analysis", json={"host_id": selected_threat})
                if response.status_code == 200:
                    st.session_state.threat_result = response.json()
                else:
                    st.error("Хост не найден в графе.")
            except Exception as e:
                st.error(f"Ошибка: {e}")
        
        if st.session_state.threat_result:
            result = st.session_state.threat_result
            
            # Кластер заражения
            threat_cluster = df_ui[df_ui['host'] == selected_threat]['cluster'].values
            if len(threat_cluster) > 0:
                st.write(f"**Кластер заражения:** {threat_cluster[0]}")
            
            # Связи
            st.write(f"**Объем трафика к первичным источникам:** {result['primary_bytes']} байт")
            st.write(f"**Топ портов (первичные источники):** {', '.join(result['primary_ports'])}")
            st.write("**Первичные связи (Кто заразился):**")
            st.write(", ".join(result['primary_connections']) if result['primary_connections'] else "Нет данных")
            
            st.write("**Вторичные связи (Распространение):**")
            st.write(", ".join(result['secondary_connections']) if result['secondary_connections'] else "Нет данных")
            
            # Карта угрозы
            st.write("### Карта угрозы")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(data=df_ui, x='x', y='y', hue='cluster', palette='viridis', ax=ax, s=10, alpha=0.6)
            
            # Выделяем аномалии (-1)
            anomalies = df_ui[df_ui['cluster'] == -1]
            if not anomalies.empty:
                ax.scatter(anomalies['x'], anomalies['y'], color='red', s=100, edgecolor='black', zorder=6, label='Аномальные узлы')
            
            threat_row = df_ui[df_ui['host'] == selected_threat]
            ax.scatter(threat_row['x'], threat_row['y'], color='red', s=200, edgecolor='black', zorder=10, label='Скомпрометирован')
            
            primary_rows = df_ui[df_ui['host'].isin(result['primary_connections'])]
            ax.scatter(primary_rows['x'], primary_rows['y'], color='orange', s=100, edgecolor='black', zorder=5, label='Первичные связи')
            
            secondary_rows = df_ui[df_ui['host'].isin(result['secondary_connections'])]
            ax.scatter(secondary_rows['x'], secondary_rows['y'], color='yellow', s=50, edgecolor='black', zorder=5, label='Вторичные связи')
            
            ax.legend(title="Вектор атаки")
            st.pyplot(fig)
    
    st.divider()
    
    # Блок расследования угроз
    st.subheader("Расследование угроз(Threat Hunter)")
    st.write("Аналитик может самостоятельно найти подозрительные узлы, не дожидаясь компрометации.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.checkbox("Брутфорс (Auth)"):
            suspicious = df_ui[df_ui['brute_force_suspicious'] == 1]
            st.write(f"Найдено: {len(suspicious)}")
            st.dataframe(suspicious[['host', 'auth_failed']].head(5))
    with col2:
        if st.checkbox("DNS Туннелирование"):
            suspicious = df_ui[df_ui['dns_suspicious'] == 1]
            st.write(f"Найдено: {len(suspicious)}")
            st.dataframe(suspicious[['host', 'dns_total_queries']].head(5))
    with col3:
        if st.checkbox("Редкие процессы (IOC)"):
            suspicious = df_ui[df_ui['rare_process_suspicious'] == 1]
            st.write(f"Найдено: {len(suspicious)}")
            st.dataframe(suspicious[['host']].head(5))
    
    if st.checkbox("Изменение активности во времени"):
        suspicious = df_ui[df_ui['time_shift_ratio'] > df_ui['time_shift_ratio'].quantile(0.95)]
        st.write(f"Хосты с аномально резким ростом трафика во 2-й половине: {len(suspicious)}")
        st.dataframe(suspicious[['host', 'time_shift_ratio']].head(5))

# Проверка журнала
st.divider()
if st.button("Показать журнал запросов (SQLite)"):
    import sqlite3
    conn = sqlite3.connect('query_logs.db')
    query_df = pd.read_sql_query("SELECT * FROM query_log ORDER BY id DESC LIMIT 10", conn)
    st.dataframe(query_df)