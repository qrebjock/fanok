import pytest

import numpy as np

from fanok.selection import adaptive_significance_threshold


@pytest.mark.parametrize(
    "w, q, offset, expected",
    [
        ([1, 2, 3, 4, 5], 0.1, 0, 1),
        ([-1, 2, -3, 4, 5], 0.1, 0, 4),
        ([-3, -2, -1, 0, 1, 2, 3], 0.1, 0, np.inf),
        ([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1, 0, 4),
        ([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.15, 0, 3),
        (
            [-1.52, 1.93, -0.76, -0.35, 1.21, -0.39, 0.08, -1.45, 0.31, -1.38],
            0.1,
            0,
            1.93,
        ),
    ],
)
def test_adaptive_significance_threshold(w, q, offset, expected):
    w = np.array(w)
    threshold = adaptive_significance_threshold(w, q, offset=offset)
    assert threshold == expected
