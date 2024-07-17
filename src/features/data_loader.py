import pandas as pd
from src.utils.util import timer_decorator, log_decorator
import logging


class CreditCardDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    @timer_decorator
    @log_decorator
    def load_data(self):
        """Carga los datos desde el archivo CSV."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"Datos cargados exitosamente. Shape: {self.data.shape}")
        except Exception as e:
            logging.error(f"Error al cargar los datos: {str(e)}")
            raise

    @log_decorator
    def show_info(self):
        """Muestra información general sobre el DataFrame."""
        if self.data is None:
            raise ValueError("Los datos no han sido cargados. Ejecute load_data() primero.")

        logging.info("Información del DataFrame:")
        self.data.info()

    @log_decorator
    def show_columns(self):
        """Muestra las columnas del DataFrame."""
        if self.data is None:
            raise ValueError("Los datos no han sido cargados. Ejecute load_data() primero.")

        logging.info("Columnas del DataFrame:")
        print(self.data.columns.tolist())

    @log_decorator
    def show_null_values(self):
        """Muestra la cantidad de valores nulos por columna."""
        if self.data is None:
            raise ValueError("Los datos no han sido cargados. Ejecute load_data() primero.")

        logging.info("Valores nulos por columna:")
        print(self.data.isnull().sum())

    def get_data(self):
        """Retorna el DataFrame cargado."""
        return self.data