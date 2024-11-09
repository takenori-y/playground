# Copyright (c) 2024 Takenori Yoshimura
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import pytest

from stats import Stats


@pytest.mark.parametrize("p", [0, 1, 6])
@pytest.mark.parametrize("mode", ["naive", "online", "batch"])
def test_stats(p: int, mode: str, m: int = 8, n: int = 10, d: int = 2) -> None:
    """Test the Stats class.

    Parameters
    ----------
    p : int >= 0
        The maximum order of moments to compute.

    mode : str
        The mode of computation.

    m : int >= 1
        The number of samples in the first dataset.

    n : int >= 1
        The number of samples in the second dataset.

    d : int >= 1
        The dimension of the data.

    """
    # Generate some random data.
    A = np.random.normal(loc=0, scale=1, size=(m, d))
    B = np.random.normal(loc=0, scale=1, size=(n, d))
    AB = np.concatenate((A, B), axis=0)

    # Compute the moments in a straightforward way.
    targets = np.empty((p + 1, d))
    targets[0] = np.full(d, m + n)
    mu = np.mean(AB, axis=0)
    if 1 <= p:
        targets[1] = mu
    if 2 <= p:
        sigma = np.std(AB, axis=0)
        targets[2] = sigma**2
    for i in range(3, p + 1):
        targets[i] = np.mean((AB - mu) ** i, axis=0) / sigma**i

    # Test the class.
    stats = Stats(p, mode=mode)
    stats.add(AB)
    moments = stats.get()
    assert np.allclose(targets, moments)

    # Test the merge method.
    stats.clear()
    stats.add(A)
    another_stats = Stats(p, mode=mode)
    another_stats.add(B)
    stats.merge(another_stats)
    moments = stats.get()
    assert np.allclose(targets, moments)