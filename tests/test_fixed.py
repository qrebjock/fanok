import numpy as np

from sklearn.datasets import make_regression

from fanok.generation import FixedKnockoffs, fixed_knockoffs, are_fixed_knockoffs_valid
from fanok.generation.fixed import fixed_knockoffs_extend_samples


def test_fixed_knockoffs():
    n, p, m = 150, 75, 20
    X, _ = make_regression(n_samples=n, n_features=p, n_informative=m)

    X_tilde = fixed_knockoffs(X, sdp_mode="equi", stack=False)
    assert are_fixed_knockoffs_valid(X, X_tilde)

    X_tilde = fixed_knockoffs(X, sdp_mode="sdp", stack=False)
    assert are_fixed_knockoffs_valid(X, X_tilde)

    fk = FixedKnockoffs()
    fk.fit(X)
    assert are_fixed_knockoffs_valid(X, fk.transform(X))

    fk = FixedKnockoffs(sdp_mode="sdp")
    fk.fit(X)
    assert are_fixed_knockoffs_valid(X, fk.transform(X))


def test_extended_fixed_knockoffs():
    n, p, m = 150, 100, 20
    X, _ = make_regression(n_samples=n, n_features=p, n_informative=m)

    X_tilde = fixed_knockoffs(X, sdp_mode="equi", stack=False)
    XX, _ = fixed_knockoffs_extend_samples(X)
    assert are_fixed_knockoffs_valid(XX, X_tilde)


def test_extend_samples():
    n, p = 30, 20
    noise_estimate = 1
    X, y = make_regression(n_samples=n, n_features=p)

    XX, yy = fixed_knockoffs_extend_samples(X, y, noise_estimate=noise_estimate)
    assert XX.shape == (2 * p, p) and yy.shape == (2 * p,)
    assert np.all(XX[n + 1 :, :] == 0)

    XX, yy = fixed_knockoffs_extend_samples(X, noise_estimate=noise_estimate)
    assert XX.shape == (2 * p, p) and yy is None
    assert np.all(XX[n + 1 :, :] == 0)
