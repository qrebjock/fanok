from sklearn.datasets import make_regression

from fanok import (
    RandomizedLowRankFactorModel,
    LowRankGaussianKnockoffs,
    SimpleKnockoffsSelector,
)
from fanok.statistics import EstimatorKnockoffStatistics


X, y, coef = make_regression(n_samples=100, n_features=200, n_informative=20, coef=True)

factor_model = RandomizedLowRankFactorModel(rank=2)
knockoffs = LowRankGaussianKnockoffs(factor_model)
statistics = EstimatorKnockoffStatistics()
selector = SimpleKnockoffsSelector(knockoffs, statistics, alpha=0.2, offset=1)
selector.fit(X, y)

fdr, power = selector.score(X, y, coef)
print(f"FDR: {fdr}, Power: {power}")
