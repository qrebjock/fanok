import numpy as np

from sklearn.base import BaseEstimator


class L1MultinomialNBStatistics(BaseEstimator):
    def __init__(
        self, remove_min: bool = True, normalize: bool = False,
    ):
        super().__init__()
        self.remove_min = remove_min
        self.normalize = normalize

    def fit(self, X, y):
        if self.remove_min:
            X = X - np.min(X)
        if self.normalize:
            X = (X.T / np.linalg.norm(X, axis=1)).T

        X_p = X[y == 1]
        X_m = X[y == 0]
        f_p = np.sum(X_p, axis=0)
        f_m = np.sum(X_m, axis=0)

        q1 = (f_p + f_m) * np.log(f_p + f_m)
        alpha = np.sum(f_p) / np.sum(f_p + f_m)
        q2 = f_p * np.log(f_p / alpha) + f_m * np.log(f_m / (1 - alpha))

        self.coef_ = q2 - q1
        return self


class L1GaussianNBStatistics(BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        X_p = X[y == 1]
        X_m = X[y == 0]

        n_p = X_p.shape[0]
        n_m = X_m.shape[0]

        mean_p = np.mean(X_p, axis=0)
        mean_m = np.mean(X_m, axis=0)
        std_p = np.std(X_p, axis=0)
        std_m = np.std(X_m, axis=0)

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        q1 = (
            -np.sum(np.square(X_p - mean_p) / 2 / np.square(std_p), axis=0)
            - np.sum(np.square(X_m - mean_m) / 2 / np.square(std_m), axis=0)
            - n_p * np.log(std_p)
            - n_m * np.log(std_m)
        )
        q2 = -np.sum(np.square(X - mean) / 2 / np.square(std), axis=0) - (
            n_p + n_m
        ) * np.log(std)

        self.coef_ = q1 - q2
        return self


class L1BernoulliNBEstimator(BaseEstimator):
    def __init__(self, remove_min: bool = True):
        super().__init__()
        self.remove_min = remove_min

    def fit(self, X, y):
        if self.remove_min:
            X = X - np.min(X)

        X_p = X[y == 1]
        X_m = X[y == 0]
        f_p = np.sum(X_p, axis=0)
        f_m = np.sum(X_m, axis=0)
        n_p = X_p.shape[0]
        n_m = X_m.shape[0]

        theta_p = f_p / n_p
        theta_m = f_m / n_m
        theta = (f_p + f_m) / (n_p + n_m)

        q1 = (
            f_p * np.log(theta_p)
            + (n_p - f_p) * np.log(1 - theta_p)
            + f_m * np.log(f_m / n_m)
            + (n_m - f_m) * np.log(1 - theta_m)
        )

        q2 = (f_p + f_m) * np.log(theta) + (n_p + n_m - f_p - f_m) * np.log(1 - theta)

        self.coef_ = q1 - q2
        return self
