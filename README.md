# FANOK: Generating knockoffs in linear time

FANOK is a Python implementation of the gaussian knockoffs framework
developed by Barber-Candès [[1]](#1) [[2]](#2).
It provides fast algorithms to generate gaussian knockoffs in high-dimensions.

# Installation

## Requirements

This package requires NumPy, Scipy and Scikit-Learn.
Use `pip install requirements.txt` to install them.

# Usage

See `examples/` to find examples.

# References

<a id="1">[1]</a>
Barber, R. F. and Candès, E. J. (2015).
Controlling the false discovery rate via knockoffs.
Ann. Statist., 43(5):2055–2085.

<a id="2">[2]</a>
Candès, Emmanuel & Fan, Yingying & Janson, Lucas & Lv, Jinchi. (2016).
Panning for Gold: Model-free Knockoffs for High-dimensional Controlled Variable Selection.
Journal of the Royal Statistical Society: Series B (Statistical Methodology).
80\. 10.1111/rssb.12265. 
