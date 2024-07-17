import logging
from functools import wraps
from time import time

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logging.info(f"Función {func.__name__} ejecutada en {end_time - start_time:.2f} segundos")
        return result
    return wrapper

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Ejecutando función: {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Función {func.__name__} completada")
        return result
    return wrapper