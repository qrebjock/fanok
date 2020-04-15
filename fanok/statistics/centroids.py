import numpy as np

from fanok.statistics import AntisymmetricKnockoffStatistics


class CentroidsL2Statistics(AntisymmetricKnockoffStatistics):
    def __init__(self, antisymmetry="difference"):
        super().__init__(antisymmetry=antisymmetry)

    def evaluate_antisymmetric(self, X, y):
        X_p, X_m = np.mean(X[y == 0, :], axis=0), np.mean(X[y == 1, :], axis=0)
        d = X_p - X_m
        return 1 / 2 * d * d


def med_w(z, w):
    threshold = np.sum(w) / 2
    arg = np.argsort(z)
    w_cum_sum = np.cumsum(w[arg])
    ind = np.argmax(w_cum_sum >= threshold)
    zeta_bar = z[arg][ind]

    if w_cum_sum[ind] > threshold:
        return zeta_bar
    else:
        return (zeta_bar + z[arg][ind + 1]) / 2


def mat_med_w(z, w):
    w_sum = np.sum(w, axis=0)
    arg = np.argsort(z, axis=0)
    w_cum_sum = np.cumsum(w[arg])
    indexes = np.argmax(w_cum_sum >= w_sum / 2, axis=0)
    zeta_bar = np.take_along_axis(z, arg, axis=0)[indexes, :]

    for a in zeta_bar:
        if a <= w_sum / 2:
            print(a, w_sum / 2)
            break

    return zeta_bar


class CentroidsL1Statistics(AntisymmetricKnockoffStatistics):
    def __init__(self, antisymmetry="difference"):
        super().__init__(antisymmetry=antisymmetry)

    def evaluate_antisymmetric(self, X, y):
        X_p = X[y == 1]
        X_m = X[y == 0]
        n_p = X_p.shape[0]
        n_m = X_m.shape[0]
        theta_p = np.median(X[y == 1], axis=0)
        theta_m = np.median(X[y == 0], axis=0)
        w = y / n_p + (1 - y) / n_m
        theta = mat_med_w(X, w)

        obj_diff = 1 / n_p * np.sum(np.abs(X_p - theta_p), axis=0) + 1 / n_m * np.sum(
            np.abs(X_m - theta_m), axis=0
        )
        obj_eq = 1 / n_p * np.sum(np.abs(X_p - theta), axis=0) + 1 / n_m * np.sum(
            np.abs(X_m - theta), axis=0
        )

        return obj_eq - obj_diff
