from sklearn.datasets import make_regression

from fanok import (
    RandomizedLowRankFactorModel,
    LowRankGaussianKnockoffs,
    KnockoffSelector,
)
from fanok.statistics import EstimatorStatistics


X, y, coef = make_regression(n_samples=100, n_features=150, n_informative=20, coef=True)

factor_model = RandomizedLowRankFactorModel(rank=20)
knockoffs = LowRankGaussianKnockoffs(factor_model)
statistics = EstimatorStatistics()
selector = KnockoffSelector(knockoffs, statistics, alpha=0.2, offset=1)
selector.fit(X, y)

fdp, power = selector.score(X, y, coef)
print(f"FDP: {fdp}, Power: {power}")
