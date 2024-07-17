import pandas as pd
from sklearn.decomposition import PCA
from src.utils.util import log_decorator
import logging


class CreditCardPCA:
    def __init__(self, data):
        self.data = data
        self.features = [col for col in data.columns if col not in ['CUST_ID', 'TENURE']]
        self.pca = None
        self.pca_data = None

    @log_decorator
    def apply_pca(self, n_components=None):
        X = self.data[self.features]
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(X)
        logging.info(f"PCA aplicado. Número de componentes: {self.pca.n_components_}")

    def get_pca_data(self, n_components=None):
        if n_components is None or n_components > self.pca.n_components_:
            n_components = self.pca.n_components_

        pca_df = pd.DataFrame(
            self.pca_data[:, :n_components],
            columns=[f'PC{i + 1}' for i in range(n_components)]
        )
        pca_df['CUST_ID'] = self.data['CUST_ID']
        pca_df['TENURE'] = self.data['TENURE']

        return pca_df