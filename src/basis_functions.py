from sklearn.base import TransformerMixin
import numpy as np


class PolynomialTransformer(TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        return self

    @staticmethod
    def transform(self, x):
        index = 0
        transformed = np.zeros((x.shape[0], (x.shape[1] + 1) * (x.shape[1] + 2) // 2), dtype=float)
        # pairwise product
        for i in range(x.shape[1]):
            for j in range(i, x.shape[1]):
                transformed.T[index] = x.T[i] * x.T[j]
                index += 1
        # repeat features
        for i in range(x.shape[1]):
            transformed.T[index] = x.T[i]
        # constant feature
        transformed.T[-1] = 1.
        return transformed
