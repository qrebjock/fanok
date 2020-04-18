from sklearn.datasets import make_regression

from fanok import (
    FixedKnockoffs,
    SimpleKnockoffsSelector,
)
from fanok.statistics import EstimatorKnockoffStatistics


X, y, coef = make_regression(n_samples=200, n_features=100, n_informative=20, coef=True)

knockoffs = FixedKnockoffs()
statistics = EstimatorKnockoffStatistics()
selector = SimpleKnockoffsSelector(knockoffs, statistics, alpha=0.2, offset=1)
selector.fit(X, y)

fdp, power = selector.score(X, y, coef)
print(f"FDP: {fdp}, Power: {power}")