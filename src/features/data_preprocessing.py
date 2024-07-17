import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.utils.util import timer_decorator, log_decorator
import logging


class NullValueHandler:
    def __init__(self, data):
        self.data = data.copy()
        self.original_dtypes = self.data.dtypes

    @log_decorator
    def show_null_info(self):
        """Muestra información sobre los valores nulos en el DataFrame."""
        null_counts = self.data.isnull().sum()
        null_percentages = 100 * self.data.isnull().sum() / len(self.data)
        null_table = pd.concat([null_counts, null_percentages], axis=1, keys=['Total', 'Porcentaje'])
        print(null_table[null_table['Total'] > 0].sort_values('Total', ascending=False))

    @timer_decorator
    @log_decorator
    def handle_nulls(self, strategy='mean', columns=None):
        """
        Maneja los valores nulos en las columnas especificadas.

        :param strategy: Estrategia de imputación ('mean', 'median', 'most_frequent', 'constant')
        :param columns: Lista de columnas a procesar. Si es None, se procesan todas las columnas con nulos.
        """
        if columns is None:
            columns = self.data.columns[self.data.isnull().any()].tolist()

        for column in columns:
            if self.data[column].dtype in ['int64', 'float64']:
                imputer = SimpleImputer(strategy=strategy)
                self.data[column] = imputer.fit_transform(self.data[[column]])
                logging.info(f"Valores nulos en {column} imputados usando {strategy}")
            else:
                logging.warning(f"La columna {column} no es numérica. No se puede aplicar la estrategia {strategy}")

    @log_decorator
    def drop_nulls(self, threshold=None):
        """
        Elimina filas con valores nulos.

        :param threshold: Número mínimo de valores no nulos para mantener una fila. Si es None, se eliminan todas las filas con al menos un nulo.
        """
        original_shape = self.data.shape
        self.data.dropna(thresh=threshold, inplace=True)
        logging.info(f"Filas eliminadas: {original_shape[0] - self.data.shape[0]}")

    @log_decorator
    def restore_dtypes(self):
        """Restaura los tipos de datos originales después de la imputación."""
        for column, dtype in self.original_dtypes.items():
            self.data[column] = self.data[column].astype(dtype)
        logging.info("Tipos de datos originales restaurados")

    def get_processed_data(self):
        """Retorna el DataFrame procesado."""
        return self.data


class CreditCardPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.original_shape = self.data.shape
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_columns.remove('TENURE')  # Asumimos que TENURE no necesita ser normalizado

    def handle_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Maneja los outliers en las columnas especificadas.

        :param columns: Lista de columnas para manejar outliers. Si es None, se usan todas las columnas numéricas.
        :param method: 'iqr' para rango intercuartil o 'zscore' para puntuación Z.
        :param threshold: Umbral para considerar un valor como outlier.
        """
        if columns is None:
            columns = self.numeric_columns

        for column in columns:
            if method == 'iqr':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)
                self.data[column] = self.data[column].clip(lower_bound, upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.data[column]))
                self.data[column] = self.data[column].mask(z_scores > threshold, self.data[column].median())

        logging.info(f"Outliers manejados en las columnas: {columns}")

    def normalize_variables(self, columns=None):
        """
        Normaliza las variables especificadas usando StandardScaler.

        :param columns: Lista de columnas para normalizar. Si es None, se usan todas las columnas numéricas excepto TENURE.
        """
        if columns is None:
            columns = [col for col in self.numeric_columns if col != 'TENURE']

        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        logging.info(f"Variables normalizadas: {columns}")

    def create_features(self):
        """
        Crea nuevas características basadas en el conocimiento del dominio y los datos existentes.
        """
        # Ratio de compras a límite de crédito
        self.data['PURCHASE_TO_CREDIT_RATIO'] = self.data['PURCHASES'] / self.data['CREDIT_LIMIT']

        # Ratio de pagos a compras
        self.data['PAYMENT_TO_PURCHASE_RATIO'] = self.data['PAYMENTS'] / self.data['PURCHASES']
        self.data['PAYMENT_TO_PURCHASE_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data['PAYMENT_TO_PURCHASE_RATIO'].fillna(0, inplace=True)

        # Frecuencia total de compras
        self.data['TOTAL_PURCHASE_FREQUENCY'] = self.data['PURCHASES_FREQUENCY'] + self.data[
            'ONEOFF_PURCHASES_FREQUENCY'] + self.data['PURCHASES_INSTALLMENTS_FREQUENCY']

        logging.info(
            "Nuevas características creadas: PURCHASE_TO_CREDIT_RATIO, PAYMENT_TO_PURCHASE_RATIO, TOTAL_PURCHASE_FREQUENCY")

    def preprocess_data(self):
        """
        Aplica todos los pasos de preprocesamiento.
        """
        self.handle_outliers()
        self.normalize_variables()
        self.create_features()

        logging.info(
            f"Preprocesamiento completado. Shape original: {self.original_shape}, Shape nuevo: {self.data.shape}")
        return self.data