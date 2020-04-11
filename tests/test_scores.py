import pytest
import numpy as np

from fanok.scores import selection_fdr, selection_power


@pytest.mark.parametrize(
    "a, b, fdr",
    [
        ([1, 2, 3, 0, 4], [1, 2, 3, 4, 5], 0),
        ([1, 2, 3, 0, 4], [1, 0, 3, 4, 0], 0.5),
        ([1, 2, 3, 0, 4], [0, 0, 0, 4, 0], 1),
    ],
)
def test_fdr(a, b, fdr):
    a, b = np.array(a), np.array(b)
    assert selection_fdr(a, b) == fdr


@pytest.mark.parametrize(
    "a, b, power",
    [
        ([1, 2, 3, 0, 4], [1, 2, 3, 4, 5], 4 / 5),
        ([1, 2, 3, 0, 4], [1, 0, 3, 4, 0], 2 / 3),
        ([1, 2, 3, 0, 4], [0, 0, 0, 4, 0], 0),
    ],
)
def test_power(a, b, power):
    a, b = np.array(a), np.array(b)
    assert selection_power(a, b) == power
