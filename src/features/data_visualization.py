import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from src.utils.util import timer_decorator, log_decorator
import logging


class DataVisualizer:
    def __init__(self, data, output_dir='visualizations'):
        self.data = data
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        try:
            plt.style.use('seaborn')
        except:
            logging.warning("El estilo 'seaborn' no está disponible. Se usará el estilo por defecto.")

    def save_figure(self, fig, filename):
        """Guarda la figura en el directorio de salida."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        logging.info(f"Figura guardada en: {filepath}")

    @timer_decorator
    @log_decorator
    def plot_distribution(self, column, bins=30):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.data[column], bins=bins, kde=True, ax=ax)
        ax.set_title(f'Distribución de {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frecuencia')
        self.save_figure(fig, f'distribution_{column}.png')
        plt.close(fig)

    @timer_decorator
    @log_decorator
    def plot_boxplot(self, column):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=self.data[column], ax=ax)
        ax.set_title(f'Boxplot de {column}')
        self.save_figure(fig, f'boxplot_{column}.png')
        plt.close(fig)

    @timer_decorator
    @log_decorator
    def plot_correlation_heatmap(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns

        corr = self.data[columns].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title('Mapa de Correlación')
        self.save_figure(fig, 'correlation_heatmap.png')
        plt.close(fig)

    @timer_decorator
    @log_decorator
    def plot_scatter(self, x, y):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=self.data, x=x, y=y, ax=ax)
        ax.set_title(f'Gráfico de dispersión: {x} vs {y}')
        self.save_figure(fig, f'scatter_{x}_vs_{y}.png')
        plt.close(fig)

    @timer_decorator
    @log_decorator
    def plot_pairplot(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns

        g = sns.pairplot(self.data[columns])
        g.fig.suptitle('Pairplot de Variables', y=1.02)
        self.save_figure(g.fig, 'pairplot.png')
        plt.close(g.fig)

    @timer_decorator
    @log_decorator
    def plot_categorical_distribution(self, column):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=self.data, x=column, ax=ax)
        ax.set_title(f'Distribución de {column}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        self.save_figure(fig, f'categorical_distribution_{column}.png')
        plt.close(fig)