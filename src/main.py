import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

class FeatureStore:
    def __init__(self):
        self.df = None
        self.X_scaled = None
        self.knn = None
        self.behavior_cols = ['total_bytes', 'avg_bytes', 'total_packets', 'avg_packets', 'avg_duration', 'entropy_dst', 'avg_packet_size', 'ratio_bytes_per_packet']
        self.extra_cols = ['dns_unique_dst', 'dns_total_queries', 'auth_unique_users', 'auth_total_events', 'auth_failed', 'proc_unique_processes', 'proc_total_events']
        self.load_and_fit()

    def load_and_fit(self):
        # 1 - загружаем все три источника данных
        df_behavior = pd.read_csv('data/host_features_v2.csv')
        df_ports = pd.read_csv('data/port_tfidf_features.csv')
        df_extra = pd.read_csv('data/full_extra_features.csv')
        
        # 2 - объединяем по хосту и заполняем пропуски нулями
        self.df = pd.merge(df_behavior, df_ports, on='host', how='inner')
        self.df = pd.merge(self.df, df_extra, on='host', how='left')
        self.df = self.df.fillna(0)

        # 3 - масштабируем поведенческие фичи
        self.scaler_behavior = StandardScaler()
        behavior_scaled = self.scaler_behavior.fit_transform(self.df[self.behavior_cols])
        
        # 4 - применяем SVD к портам (убираем шум, сокращаем до 50 измерений)
        port_cols = [col for col in self.df.columns if col.startswith('port_')]
        svd = TruncatedSVD(n_components=50, random_state=42)
        ports_svd = svd.fit_transform(self.df[port_cols])
        
        # 5 - масштабируем дополнительные фичи (DNS, Auth, Proc)
        self.scaler_extra = StandardScaler()
        extra_scaled = self.scaler_extra.fit_transform(self.df[self.extra_cols])

        # 6 - объединяем все три матрицы
        X_combined = np.hstack((behavior_scaled, ports_svd, extra_scaled))
        
        # 7 - финальная нормализация (чтобы порты не доминировали)
        self.scaler_combined = StandardScaler()
        self.X_scaled = self.scaler_combined.fit_transform(X_combined)
        
        # 8 - обучаем KNN
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn.fit(self.X_scaled)

    def get_similar(self, host_id, top_n=5):
        if host_id not in self.df['host'].values:
            raise HTTPException(status_code=404, detail="Host not found")
        
        idx = self.df.index[self.df['host'] == host_id][0]
        distances, indices = self.knn.kneighbors(self.X_scaled[idx].reshape(1, -1), n_neighbors=top_n + 1)
        
        port_cols = [col for col in self.df.columns if col.startswith('port_')]
        
        results = []
        query_host_data = self.df.iloc[idx]
        
        for i in range(1, top_n + 1):
            similar_idx = indices[0][i]
            similar_host = self.df.iloc[similar_idx]
            similarity = round(1 - distances[0][i], 2)
            
            # 1 - общие порты
            ports1 = set(query_host_data[port_cols][lambda x: x > 0].index)
            ports2 = set(similar_host[port_cols][lambda x: x > 0].index)
            common = [p.replace('port_', '') for p in list(ports1.intersection(ports2))[:5]]
            
            # 2 - общие поведенческие черты
            behavior_traits = []
            if abs(query_host_data['total_bytes'] - similar_host['total_bytes']) < 1.5:
                behavior_traits.append("объем трафика")
            if abs(query_host_data['entropy_dst'] - similar_host['entropy_dst']) < 0.5:
                behavior_traits.append("разнообразие получателей")
            if abs(query_host_data['avg_packet_size'] - similar_host['avg_packet_size']) < 1.0:
                behavior_traits.append("размер пакетов")

            # 3 - новые черты (DNS, Auth, Proc)
            if abs(query_host_data['auth_unique_users'] - similar_host['auth_unique_users']) < 2:
                behavior_traits.append("одинаковые пользователи")
            if abs(query_host_data['dns_total_queries'] - similar_host['dns_total_queries']) < 50:
                behavior_traits.append("схожая DNS-активность")
                
            results.append({
                "host": similar_host['host'],
                "similarity": similarity,
                "common_ports": common,
                "behavior_traits": behavior_traits[:3] #берем до 3 черт
            })
        
        return results

class QueryLogger:
    def __init__(self, db_path='query_logs.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query_host TEXT,
                result_json TEXT
            )
        ''')
        self.conn.commit()

    def log_query(self, host, result):
        timestamp = datetime.datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO query_log (timestamp, query_host, result_json) VALUES (?, ?, ?)",
                       (timestamp, host, json.dumps(result)))
        self.conn.commit()

# Класс для анализа угроз
class ThreatAnalyzer:
    def __init__(self):
        with open('data/network_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        self.df_extra = pd.read_csv('data/full_extra_features.csv')

    def analyze(self, compromised_host):
        if compromised_host not in self.graph:
            raise HTTPException(status_code=404, detail="Host not found in graph")
        
        # Первичные связи
        primary_edges = self.graph.get(compromised_host, {})
        primary = list(primary_edges.keys())
        
        # Считаем объем данных и топ-портов
        primary_bytes = sum(edge['bytes'] for edge in primary_edges.values())
        all_ports = set()
        for edge in primary_edges.values():
            all_ports.update(edge['ports'])
        primary_ports = [str(p) for p in list(all_ports)[:5]]
        
        # Вторичные связи
        secondary = set()
        for host in primary:
            if host in self.graph:
                secondary.update(self.graph[host].keys())
        secondary.discard(compromised_host)
        secondary = list(secondary)[:50]
        
        # Данные о процессах и пользователях
        target_info = self.df_extra[self.df_extra['host'] == compromised_host]
        processes = int(target_info['proc_unique_processes'].values[0]) if not target_info.empty else 0
        users = int(target_info['auth_unique_users'].values[0]) if not target_info.empty else 0
            
        return {
            "compromised_host": compromised_host,
            "primary_connections": primary[:20],
            "secondary_connections": secondary,
            "primary_bytes": primary_bytes,
            "primary_ports": primary_ports,
            "processes_on_host": processes,
            "users_on_host": users
        }

# инициализация (единожды)
app = FastAPI(title="Network Traffic Similarity API")
store = FeatureStore()
logger = QueryLogger()
threat_analyzer = ThreatAnalyzer()

# эндпоинты
class QueryRequest(BaseModel):
    host_id: str

class SimilarHost(BaseModel):
    host: str
    similarity: float
    common_ports: list[str]
    behavior_traits: list[str] = []

class SimilarResponse(BaseModel):
    query: str
    most_similar: list[SimilarHost]

class ThreatRequest(BaseModel):
    host_id: str

@app.post("/find_similar", response_model=SimilarResponse)
def find_similar(request: QueryRequest):
    results = store.get_similar(request.host_id)
    logger.log_query(request.host_id, results)
    return SimilarResponse(
        query=request.host_id,
        most_similar=[SimilarHost(**res) for res in results]
    )

@app.post("/threat_analysis")
def threat_analysis(request: ThreatRequest):
    result = threat_analyzer.analyze(request.host_id)
    logger.log_query(request.host_id, result)
    return result

@app.get("/")
def root():
    return {"message": "API is running"}