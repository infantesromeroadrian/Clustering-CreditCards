import os

# Rutas de archivos
DATA_DIR = os.path.join('..', 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_data', 'creditcards.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data', 'creditcards_preprocessed.csv')
CLUSTERED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data', 'creditcards_clusters.csv')

# Rutas de modelos
MODEL_DIR = os.path.join('..', 'models')
PCA_MODEL_PATH = os.path.join(MODEL_DIR, 'pca_model.pkl')
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, 'kmeans_model.pkl')

# Configuraciones de visualización
VISUALIZATION_DIR = 'visualizations'

# Configuraciones de clustering
N_CLUSTERS = 3
RANDOM_STATE = 42

# Configuraciones de PCA
N_COMPONENTS = 3