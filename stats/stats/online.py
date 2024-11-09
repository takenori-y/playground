# Copyright (c) 2024 Takenori Yoshimura
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
from typing import Optional

import numpy as np

from .base import BaseStats
from .utils import power


class OnlineStats(BaseStats):
    """A class for calculating the moments of data in an online manner."""

    def __init__(self, p: int) -> None:
        """Initialize the class.

        Parameters
        ----------
        p : int >= 0
            The maximum order of the moments to be calculated.

        """
        if p <= -1 or not isinstance(p, int):
            raise ValueError(
                "The maximum order of the moments must be a non-negative integer."
            )
        self.p = p
        self.ps = np.arange(p + 1)
        self.buffer: Optional[np.ndarray] = None

    def get(self) -> np.ndarray:
        """Return the moments of the data.

        Returns
        -------
        out : np.ndarray [shape=(p + 1, d)]
            The moments of the data.

        """
        if self.buffer is None:
            return np.empty((0, 0))

        moments = self.buffer.copy()
        if 2 <= self.p:
            moments[2:] /= moments[0]
            sigma_powers = power(np.sqrt(moments[2]), self.ps[3:])
            moments[3:] /= sigma_powers
        return moments

    def add(self, x: np.ndarray) -> None:
        """Add a new data point(s) to the statistics.

        Parameters
        ----------
        x : np.ndarray [shape=(d,) or (n, d)]
            The data point(s) to add.

        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("The input must be a 1D or 2D array.")

        d = x.shape[1]
        if self.buffer is None:
            self.buffer = np.zeros((self.p + 1, d))
        elif d != self.buffer.shape[1]:
            raise ValueError("The dimensionality of the input is inconsistent.")

        inputs = np.zeros_like(self.buffer)
        inputs[0] = 1
        for data_point in x:
            if 1 <= self.p:
                inputs[1] = data_point
            self.buffer = self._calc_unnormalized_n_moments(inputs)

    def merge(self, stats: BaseStats) -> None:
        """Merge the statistics with another set of statistics.

        Parameters
        ----------
        stats : BaseStats
            The statistics to merge with.

        """
        if not isinstance(stats, type(self)):
            raise ValueError("The input must be an object of the same class.")

        if stats.buffer is None:
            return

        if self.buffer is None:
            self.buffer = stats.buffer.copy()
            return

        self.buffer = self._calc_unnormalized_n_moments(stats.buffer)

    def clear(self) -> None:
        """Clear the sufficient statistics."""
        self.buffer = None

    def _calc_unnormalized_n_moments(self, x: np.ndarray) -> np.ndarray:
        """Calculate the unnormalized moments of the data.

        Parameters
        ----------
        x : np.ndarray [shape=(p + 1, d)]
            The sum of the powers of the differences between the data points and the
            current mean.

        Returns
        -------
        out : np.ndarray [shape=(p + 1, d)]
            The unnormalized moments of the data. Note that the moments of order two or
            higher are not averaged by the number of data points.

        """
        assert self.buffer is not None
        moments = np.zeros_like(x)

        # 0th-order moment.
        m = self.buffer[0]
        n = x[0]
        moments[0] = m + n
        if self.p == 0:
            return moments

        mn = m * n
        ms = power(m, self.ps)
        ns = power(-n, self.ps)
        z = power(moments[0], -self.ps)

        # Higher-order moments.
        delta = x[1] - self.buffer[1]
        delta_powers = power(delta, self.ps)
        for q in reversed(range(2, self.p + 1)):
            q1 = q - 1
            term1 = x[q] + self.buffer[q]
            term2 = np.sum(
                [
                    (math.comb(q, i) * z[i] * delta_powers[i])
                    * (ms[i] * x[q - i] + ns[i] * self.buffer[q - i])
                    for i in range(1, q1)
                ],
                axis=0,
            )
            c = np.sum(np.flip(ms[:q1], axis=0) * ns[:q1], axis=0)
            term3 = c * mn * z[q1] * delta_powers[q]
            moments[q] = term1 + term2 + term3

        # 1st-order moment.
        moments[1] = z[1] * n * delta + self.buffer[1]

        return moments
