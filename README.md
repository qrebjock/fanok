# FANOK: Generating knockoffs in linear time

FANOK is a Python implementation of the Gaussian knockoffs framework
developed by Barber-Candès [[1]](#1) [[2]](#2).
It provides fast algorithms based on coordinate ascent to generate Gaussian knockoffs in high dimensions.

## Installation

### Requirements

This package requires NumPy, Scipy, Scikit-Learn and CVXPY.
Use `pip install requirements.txt` to install them.

### Installation

`pip install fanok`

## Usage

Here is a minimal usage example:
```python
from sklearn.datasets import make_regression
from fanok import GaussianKnockoffs, KnockoffSelector
from fanok.statistics import EstimatorKnockoffStatistics

X, y, coef = make_regression(n_samples=100, n_features=150, n_informative=20, coef=True)

knockoffs = GaussianKnockoffs()
statistics = EstimatorKnockoffStatistics()
selector = KnockoffSelector(knockoffs, statistics, alpha=0.2, offset=1)
selector.fit(X, y)

fdp, power = selector.score(X, y, coef)
print(f"FDP: {fdp}, Power: {power}")
```

See the folder `examples/` for more illustrations,
and in particular with fixed and low-rank knockoffs.

## References

<a id="1">[1]</a>
Barber, R. F. and Candès, E. J. (2015).
Controlling the false discovery rate via knockoffs.
Ann. Statist., 43(5):2055–2085.

<a id="2">[2]</a>
Candès, Emmanuel & Fan, Yingying & Janson, Lucas & Lv, Jinchi. (2016).
Panning for Gold: Model-free Knockoffs for High-dimensional Controlled Variable Selection.
Journal of the Royal Statistical Society: Series B (Statistical Methodology).
80\. 10.1111/rssb.12265. 
