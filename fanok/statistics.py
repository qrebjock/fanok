import numpy as np

from sklearn.linear_model import Lasso, LassoCV, LassoLars

from naive_feature_selection.sparse_naive_bayes import sparse_naive_bayes


def antisymmetric_knockoff_statistics(
    z: np.ndarray, z_tilde: np.ndarray = None, mode: str = "difference"
):
    if z_tilde is None:
        z, z_tilde = np.split(z, 2)

    if mode == "max":
        return np.maximum(z, z_tilde) * np.sign(z - z_tilde)
    elif mode == "difference":
        return np.abs(z) - np.abs(z_tilde)
    elif mode == "log":
        return np.log(z / z_tilde)
    else:
        raise ValueError(
            f"Argument 'mode' can be 'max', 'difference' or 'log'. Found {mode}"
        )


class KnockoffStatistics:
    def __init__(self):
        pass

    def evaluate(self, X, X_tilde, y):
        pass

    def __call__(self, X, X_tilde, y):
        return self.evaluate(X, X_tilde, y)


class PlainKnockoffStatistics(KnockoffStatistics):
    def __init__(self):
        super().__init__()

    def evaluate(self, X, X_tilde, y):
        pass


class AntisymmetricKnockoffStatistics(KnockoffStatistics):
    def __init__(self, antisymmetry="difference"):
        super().__init__()
        self.antisymmetry = antisymmetry

    def evaluate(self, X, X_tilde, y):
        return antisymmetric_knockoff_statistics(
            self.evaluate_antisymmetric(X, X_tilde, y), mode=self.antisymmetry
        )

    def evaluate_antisymmetric(self, X, X_tilde, y):
        pass


class EstimatorKnockoffStatistics(AntisymmetricKnockoffStatistics):
    def __init__(
        self, antisymmetry="difference", estimator=None, absolute: bool = True
    ):
        super().__init__(antisymmetry=antisymmetry)

        if estimator is None:
            estimator = LassoCV(cv=3)
        self.estimator = estimator
        self.absolute = absolute

    def evaluate_antisymmetric(self, X, X_tilde, y):
        stats = self.estimator.fit(np.hstack((X, X_tilde)), y).coef_
        if self.absolute:
            stats = np.abs(stats)
        return stats


class LarsStatistics(AntisymmetricKnockoffStatistics):
    """
    Efficiently computes z = sup{lambda | beta(lambda) != 0}.
    where beta(lambda) is the coefficient vector of a Lasso model penalized
    by the coefficient lambda.
    Advantages:
        - Finds the exact solution (rather than trying a finite number of lambdas)
        - No need to tune the search space
    Can however be slow if the matrix [X, X_tilde] is large.
    """

    def __init__(
        self, antisymmetry="difference", normalize: bool = False, lambdas=None
    ):
        super().__init__(antisymmetry=antisymmetry)
        self.normalize = normalize
        self.lambdas = lambdas

    def evaluate_antisymmetric(self, X, X_tilde, y):
        if self.lambdas is None:
            lars = LassoLars(
                alpha=1e-15, normalize=self.normalize, max_iter=4 * X.shape[1]
            )
            lars.fit(np.hstack((X, X_tilde)), y)
            return lars.alphas_[np.argmax(lars.coef_path_.astype(bool), axis=1) - 1]
        else:
            if self.lambdas is None:
                lambdas = np.logspace(-3, 3, 100)
            coefficients_list = np.zeros(X.shape[1])
            for lam in lambdas:
                clf = Lasso(alpha=lam)
                clf.fit(X, y)
                coefficients_list = np.maximum(
                    coefficients_list, lam * clf.coef_.astype(bool)
                )
            return coefficients_list


class L1MultinomialNBStatistics(AntisymmetricKnockoffStatistics):
    def __init__(
        self,
        antisymmetry="difference",
        remove_min: bool = True,
        normalize: bool = False,
    ):
        super().__init__(antisymmetry=antisymmetry)
        self.remove_min = remove_min
        self.normalize = normalize

    def evaluate_antisymmetric(self, X, X_tilde, y):
        if self.remove_min:
            X = X - np.min(X)
        if self.normalize:
            X = (X.T / np.linalg.norm(X, axis=1)).T
        x_p = X[y == 1]
        X_m = X[y == 0]
        f_p = np.sum(x_p, axis=0)
        f_m = np.sum(X_m, axis=0)

        q1 = (f_p + f_m) * np.log(f_p + f_m)
        alpha = np.sum(f_p) / np.sum(f_p + f_m)
        q2 = f_p * np.log(f_p / alpha) + f_m * np.log(f_m / (1 - alpha))

        return q2 - q1


class L1GaussianNBStatistics(AntisymmetricKnockoffStatistics):
    def __init__(self, antisymmetry="difference"):
        super().__init__(antisymmetry=antisymmetry)

    def evaluate_antisymmetric(self, X, y):
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

        return q1 - q2


class L1BernoulliNBStatistics(AntisymmetricKnockoffStatistics):
    def __init__(self, antisymmetry="difference", remove_min: bool = True):
        super().__init__(antisymmetry=antisymmetry)
        self.remove_min = remove_min

    def evaluate_antisymmetric(self, X, y):
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

        return q1 - q2


# Centroids


class CentroidsL2Statistics(AntisymmetricKnockoffStatistics):
    def __init__(self, antisymmetry="difference"):
        super().__init__(antisymmetry=antisymmetry)

    def evaluate_antisymmetric(self, X, y):
        x_p, X_m = np.mean(X[y == 0, :], axis=0), np.mean(X[y == 1, :], axis=0)
        d = x_p - X_m
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
