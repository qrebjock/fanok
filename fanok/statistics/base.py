import numpy as np

from sklearn.linear_model import LassoCV


class KnockoffStatistics:
    """
    Abstraction of flip-sign knockoff statistics.
    """

    def __init__(self):
        pass

    def evaluate(self, X, X_tilde, y):
        pass

    def __call__(self, X, X_tilde, y):
        return self.evaluate(X, X_tilde, y)


class PlainKnockoffStatistics(KnockoffStatistics):
    """
    Flip-sign statistics directly computed from the concatenated
    feature matrix [X, \tilde{X}], without the use of an
    antisymmetric function.
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, X, X_tilde, y):
        pass


def antisymmetric_knockoff_statistics(
    z: np.ndarray, z_tilde: np.ndarray = None, mode: str = "difference"
):
    """
    Applies an anti-symmetric function to pairs of original and knockoff
    statistics. This is a natural way to obtain statistics satisfying
    the flip sign property that is required to control the FDR.
    """
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


class AntisymmetricStatistics(KnockoffStatistics):
    """
    Flip-sign knockoff statistics computed in two steps,
    through and antisymmetric function.
    """

    def __init__(self, antisymmetry="difference"):
        super().__init__()
        self.antisymmetry = antisymmetry

    def evaluate(self, X, X_tilde, y):
        return antisymmetric_knockoff_statistics(
            self.evaluate_antisymmetric(X, X_tilde, y), mode=self.antisymmetry
        )

    def evaluate_antisymmetric(self, X, X_tilde, y):
        pass


class EstimatorStatistics(AntisymmetricStatistics):
    """
    Flip-sign knockoff statistics
    """

    def __init__(
        self, estimator=None, antisymmetry="difference", absolute: bool = True
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
