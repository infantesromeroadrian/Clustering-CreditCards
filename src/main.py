import pandas as pd
import joblib
from src.utils.config import *
from src.features.data_loader import CreditCardDataLoader
from src.features.data_preprocessing import NullValueHandler, CreditCardPreprocessor
from src.features.data_visualization import DataVisualizer
from src.features.pca_analysis import CreditCardPCA
from src.models.clustering import CreditCardClusterPrep, CreditCardClusterAnalyzer
import logging


def main():
    # Cargar datos
    loader = CreditCardDataLoader(RAW_DATA_PATH)
    loader.load_data()
    data = loader.get_data()

    # Preprocesamiento
    null_handler = NullValueHandler(data)
    null_handler.handle_nulls(strategy='median', columns=['CREDIT_LIMIT', 'MINIMUM_PAYMENTS'])
    null_handler.restore_dtypes()
    processed_df = null_handler.get_processed_data()

    preprocessor = CreditCardPreprocessor(processed_df)
    processed_df = preprocessor.preprocess_data()

    # Guardar datos preprocesados
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logging.info(f"Datos preprocesados guardados en {PROCESSED_DATA_PATH}")

    # Visualización
    visualizer = DataVisualizer(processed_df, VISUALIZATION_DIR)
    visualizer.plot_distribution('BALANCE')
    visualizer.plot_boxplot('CREDIT_LIMIT')
    visualizer.plot_correlation_heatmap()
    visualizer.plot_scatter('BALANCE', 'CREDIT_LIMIT')
    visualizer.plot_pairplot(['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS'])
    visualizer.plot_categorical_distribution('TENURE')

    # Análisis PCA
    pca_analyzer = CreditCardPCA(processed_df)
    pca_analyzer.apply_pca(n_components=N_COMPONENTS)
    pca_data = pca_analyzer.get_pca_data()

    # Clustering
    cluster_prep = CreditCardClusterPrep(pca_data)
    prepared_data = cluster_prep.prepare_data()
    cluster_prep.elbow_method()
    cluster_prep.silhouette_analysis()
    elbow_suggestion, silhouette_suggestion = cluster_prep.recommend_optimal_clusters()

    analyzer = CreditCardClusterAnalyzer(pca_data, processed_df)
    analyzer.perform_clustering([elbow_suggestion, silhouette_suggestion])

    for n_clusters in [elbow_suggestion, silhouette_suggestion]:
        analyzer.visualize_clusters_2d(n_clusters)
        analyzer.visualize_clusters_3d(n_clusters)
        analyzer.characterize_clusters(n_clusters)
        analyzer.map_to_original_features(n_clusters)
        analyzer.analyze_client_profiles(n_clusters)
        analyzer.validate_clustering()
        analyzer.interpret_results(n_clusters)

    # Seleccionar el mejor número de clusters (puedes ajustar esto según tus criterios)
    best_n_clusters = silhouette_suggestion

    # Guardar modelos
    joblib.dump(pca_analyzer.pca, PCA_MODEL_PATH)
    logging.info(f"Modelo PCA guardado en {PCA_MODEL_PATH}")

    optimal_kmeans_model = KMeans(n_clusters=best_n_clusters, random_state=RANDOM_STATE)
    optimal_kmeans_model.fit(pca_data.drop(['CUST_ID', 'TENURE'], axis=1, errors='ignore'))
    joblib.dump(optimal_kmeans_model, KMEANS_MODEL_PATH)
    logging.info(f"Modelo K-Means guardado en {KMEANS_MODEL_PATH}")

    # Guardar resultados finales
    final_data = pca_data.copy()
    final_data['Cluster'] = optimal_kmeans_model.labels_
    final_data.to_csv(CLUSTERED_DATA_PATH, index=False)
    logging.info(f"Datos con clusters asignados guardados en {CLUSTERED_DATA_PATH}")

    logging.info("Análisis completo. Todos los resultados han sido guardados.")


if __name__ == "__main__":
    main()