import numpy as np

from sklearn.base import BaseEstimator


def mat_med_w(z, w):
    w_sum = np.sum(w, axis=0)
    arg = np.argsort(z, axis=0)
    w_cum_sum = np.cumsum(w[arg])
    indexes = np.argmax(w_cum_sum >= w_sum / 2, axis=0)
    zeta_bar = np.take_along_axis(z, arg, axis=0)[indexes, :]

    # TODO: Handle special case
    # for a in zeta_bar:
    #     if a <= w_sum / 2:
    #         print(a, w_sum / 2)
    #         break

    return zeta_bar


class CentroidsEstimator(BaseEstimator):
    def __init__(self, norm: str = "l1"):
        self.norm = norm.lower()
        if self.norm != "l1" and self.norm != "l2":
            raise ValueError(f"Norm may be either L1 or L2, found {norm}")
        super().__init__()

    def fit(self, X, y):
        if self.norm == "l1":
            X_p, X_m = np.mean(X[y == 0, :], axis=0), np.mean(X[y == 1, :], axis=0)
            d = X_p - X_m
            self.coef_ = 1 / 2 * d * d
        else:
            X_p = X[y == 1]
            X_m = X[y == 0]
            n_p = X_p.shape[0]
            n_m = X_m.shape[0]
            theta_p = np.median(X[y == 1], axis=0)
            theta_m = np.median(X[y == 0], axis=0)
            w = y / n_p + (1 - y) / n_m
            theta = mat_med_w(X, w)

            obj_diff = (
                np.sum(np.abs(X_p - theta_p), axis=0) / n_p
                + np.sum(np.abs(X_m - theta_m), axis=0) / n_m
            )
            obj_eq = (
                np.sum(np.abs(X_p - theta), axis=0) / n_p
                + np.sum(np.abs(X_m - theta), axis=0) / n_m
            )

            self.coef_ = obj_eq - obj_diff

        return self
