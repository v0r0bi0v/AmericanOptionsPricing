import numpy as np
from sklearn.base import RegressorMixin
from abc import abstractmethod


class AbstractRegressor(RegressorMixin):
    @abstractmethod
    def is_fitted(self):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, x, y):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError()


class RegressorLinearWithPseudoInverse(AbstractRegressor):
    def __init__(
            self,
            alpha: float = 1e-6,
            tolerance_svd: float = 1e-4
    ):
        self.alpha = alpha
        self.tolerance_svd = tolerance_svd
        self.weights = None

    def fit(self, x, y):
        regularization = np.eye(x.shape[1], dtype=float) * self.alpha
        inv = np.linalg.pinv((x.T @ x + regularization), rcond=1e-4)
        self.weights = inv @ x.T @ y

    def predict(self, x):
        if self.weights is None:
            raise AttributeError("You have to fit the model before predicting")
        return (x @ self.weights).reshape(-1)

    def is_fitted(self):
        return self.weights is not None
