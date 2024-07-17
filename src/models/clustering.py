import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils.util import log_decorator
import logging


class CreditCardClusterPrep:
    def __init__(self, pca_data):
        self.pca_data = pca_data
        self.prepared_data = None
        self.kmeans_results = {}

    def prepare_data(self):
        pc_columns = [col for col in self.pca_data.columns if col.startswith('PC')]
        self.prepared_data = self.pca_data[pc_columns + ['CUST_ID', 'TENURE']]
        logging.info(f"Datos preparados con {len(pc_columns)} componentes principales.")
        return self.prepared_data

    def elbow_method(self, max_clusters=10):
        """
        Implementa el método del codo para K-means.
        """
        inertias = []
        k_values = range(1, max_clusters + 1)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.prepared_data.drop(['CUST_ID', 'TENURE'], axis=1))
            inertias.append(kmeans.inertia_)
            self.kmeans_results[k] = kmeans

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'bo-')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo para K-means')
        plt.show()

    def silhouette_analysis(self, max_clusters=10):
        """
        Calcula y grafica el coeficiente de silueta para diferentes números de clusters.
        """
        silhouette_scores = []
        k_values = range(2, max_clusters + 1)  # Silhouette score no está definido para k=1

        for k in k_values:
            kmeans = self.kmeans_results.get(k) or KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.prepared_data.drop(['CUST_ID', 'TENURE'], axis=1))
            score = silhouette_score(self.prepared_data.drop(['CUST_ID', 'TENURE'], axis=1), labels)
            silhouette_scores.append(score)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, 'bo-')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Coeficiente de Silueta')
        plt.title('Análisis de Silueta para K-means')
        plt.show()

    def recommend_optimal_clusters(self):
        """
        Recomienda un número óptimo de clusters basado en el método del codo y el análisis de silueta.
        """
        inertias = [kmeans.inertia_ for kmeans in self.kmeans_results.values()]
        inertia_changes = np.diff(inertias)
        elbow_point = np.argmin(inertia_changes) + 2

        silhouette_scores = [silhouette_score(self.prepared_data.drop(['CUST_ID', 'TENURE'], axis=1),
                                              kmeans.labels_) for k, kmeans in self.kmeans_results.items() if k > 1]
        max_silhouette_point = np.argmax(silhouette_scores) + 2

        logging.info(f"Método del codo sugiere {elbow_point} clusters")
        logging.info(f"Análisis de silueta sugiere {max_silhouette_point} clusters")

        return elbow_point, max_silhouette_point


class CreditCardClusterAnalyzer:
    def __init__(self, pca_data, original_data):
        self.pca_data = pca_data
        self.original_data = original_data
        self.cluster_results = {}
        self.silhouette_scores = {}

    @log_decorator
    def perform_clustering(self, n_clusters_list=[2, 3]):
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(self.pca_data.drop(['CUST_ID', 'TENURE'], axis=1, errors='ignore'))
            self.cluster_results[n_clusters] = labels
            self.silhouette_scores[n_clusters] = silhouette_score(
                self.pca_data.drop(['CUST_ID', 'TENURE'], axis=1, errors='ignore'), labels
            )
            logging.info(f"Clustering realizado con {n_clusters} clusters. "
                         f"Silhouette Score: {self.silhouette_scores[n_clusters]:.4f}")

    def visualize_clusters_2d(self, n_clusters):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_data['PC1'], self.pca_data['PC2'],
                              c=self.cluster_results[n_clusters], cmap='viridis')
        plt.title(f'Visualización de Clusters ({n_clusters} clusters)')
        plt.xlabel('Primer Componente Principal')
        plt.ylabel('Segundo Componente Principal')
        plt.colorbar(scatter)
        plt.show()

    def visualize_clusters_3d(self, n_clusters):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.pca_data['PC1'], self.pca_data['PC2'], self.pca_data['PC3'],
                             c=self.cluster_results[n_clusters], cmap='viridis')
        ax.set_title(f'Visualización 3D de Clusters ({n_clusters} clusters)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.colorbar(scatter)
        plt.show()

    def characterize_clusters(self, n_clusters):
        self.pca_data['Cluster'] = self.cluster_results[n_clusters]

        numeric_columns = self.pca_data.select_dtypes(include=[np.number]).columns

        cluster_stats = self.pca_data.groupby('Cluster')[numeric_columns].agg(['mean', 'median', 'std'])
        print(f"Estadísticas de clusters para {n_clusters} clusters:")
        print(cluster_stats)
        return cluster_stats

    def map_to_original_features(self, n_clusters):
        self.original_data['Cluster'] = self.cluster_results[n_clusters]

        numeric_columns = self.original_data.select_dtypes(include=[np.number]).columns

        original_cluster_stats = self.original_data.groupby('Cluster')[numeric_columns].agg(['mean', 'median', 'std'])
        print(f"Estadísticas de características originales para {n_clusters} clusters:")
        print(original_cluster_stats)
        return original_cluster_stats

    def analyze_client_profiles(self, n_clusters):
        profiles = {}
        for cluster in range(n_clusters):
            cluster_data = self.original_data[self.original_data['Cluster'] == cluster]
            profile = {
                'size': len(cluster_data),
                'avg_balance': cluster_data['BALANCE'].mean(),
                'avg_purchases': cluster_data['PURCHASES'].mean(),
                'avg_credit_limit': cluster_data['CREDIT_LIMIT'].mean(),
                'common_tenure': cluster_data['TENURE'].mode().values[0]
            }
            profiles[cluster] = profile
            print(f"Perfil del Cluster {cluster}:")
            print(profile)
            print("\n")
        return profiles

    def validate_clustering(self):
        for n_clusters, score in self.silhouette_scores.items():
            print(f"Coeficiente de silueta para {n_clusters} clusters: {score:.4f}")

    def interpret_results(self, n_clusters):
        profiles = self.analyze_client_profiles(n_clusters)
        for cluster, profile in profiles.items():
            print(f"Interpretación del Cluster {cluster}:")
            if profile['avg_balance'] > self.original_data['BALANCE'].mean():
                print("- Clientes con saldo alto")
            else:
                print("- Clientes con saldo bajo")
            if profile['avg_purchases'] > self.original_data['PURCHASES'].mean():
                print("- Compradores frecuentes")
            else:
                print("- Compradores poco frecuentes")
            print(f"- Límite de crédito promedio: ${profile['avg_credit_limit']:.2f}")
            print(f"- Antigüedad típica: {profile['common_tenure']} meses")
            print("\nEstrategias sugeridas:")
            if profile['avg_purchases'] > self.original_data['PURCHASES'].mean():
                print("- Ofrecer programas de recompensas para mantener la lealtad")
            else:
                print("- Incentivar el uso de la tarjeta con promociones especiales")
            if profile['avg_balance'] > self.original_data['BALANCE'].mean():
                print("- Ofrecer productos de ahorro o inversión")
            else:
                print("- Considerar aumentar el límite de crédito para clientes confiables")
            print("\n")