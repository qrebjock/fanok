import numpy as np

from sklearn.linear_model import Lasso, LassoLars

from fanok.statistics import AntisymmetricKnockoffStatistics


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
