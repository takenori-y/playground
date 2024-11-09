# Copyright (c) 2024 Takenori Yoshimura
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import multiprocessing
from typing import Optional

import numpy as np

from .base import BaseStats
from .online import OnlineStats


class BatchedOnlineStats(BaseStats):
    """A class for computing statistics in a batched online fashion."""

    def __init__(self, p: int, batch_size: int = 1):
        """Initialize the class.

        Parameters
        ----------
        p : int >= 0
            The maximum order of the moments to be calculated.

        batch_size : int >= 1
            The number of samples to be processed at once.

        """
        if batch_size <= 0:
            raise ValueError("The batch size must be a positive integer.")
        self.p = p
        self.b = batch_size
        self.stats = [OnlineStats(p) for _ in range(batch_size)]

    def get(self) -> np.ndarray:
        """Return the moments of the data.

        Returns
        -------
        out : np.ndarray [shape=(p + 1, d)]
            The moments of the data.

        """
        merged_stats = OnlineStats(self.p)
        for stats in self.stats:
            merged_stats.merge(stats)
        return merged_stats.get()

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

        with multiprocessing.Pool() as pool:
            buffers = pool.starmap(
                self._calc_next_buffer, [(i, x) for i in range(self.b)]
            )
        for i in range(self.b):
            self.stats[i].buffer = buffers[i]

    def sub(self, x: np.ndarray) -> None:
        """Subtract a data point(s) from the statistics.

        Parameters
        ----------
        x : np.ndarray [shape=(d,) or (n, d)]
            The data point(s) to subtract.

        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("The input must be a 1D or 2D array.")

        with multiprocessing.Pool() as pool:
            buffers = pool.starmap(
                self._calc_prev_buffer, [(i, x) for i in range(self.b)]
            )
        for i in range(self.b):
            self.stats[i].buffer = buffers[i]

    def merge(self, stats: BaseStats) -> None:
        """Merge the statistics with another set of statistics.

        Parameters
        ----------
        stats : BaseStats
            The statistics to merge with.

        """
        if not isinstance(stats, type(self)):
            raise ValueError("The input must be an object of the same class.")
        if len(self.stats) != len(stats.stats):
            raise ValueError("The batch sizes must be the same.")

        for i in range(self.b):
            self.stats[i].merge(stats.stats[i])

    def purge(self, stats: BaseStats) -> None:
        """Purge the statistics with another set of statistics.

        Parameters
        ----------
        stats : BaseStats
            The statistics to purge with.

        """
        if not isinstance(stats, type(self)):
            raise ValueError("The input must be an object of the same class.")
        if len(self.stats) != len(stats.stats):
            raise ValueError("The batch sizes must be the same.")

        for i in range(self.b):
            self.stats[i].purge(stats.stats[i])

    def clear(self) -> None:
        """Clear the sufficient statistics."""
        for stats in self.stats:
            stats.clear()

    def _calc_next_buffer(self, i: int, x: np.ndarray) -> Optional[np.ndarray]:
        """Calculate the updated buffer for the i-th batch.

        Parameters
        ----------
        i : int >= 0
            The index of the batch.

        x : np.ndarray [shape=(n, d)]
            The data points.

        Returns
        -------
        out : np.ndarray [shape=(p + 1, d)]
            The updated buffer.

        """
        if len(x) <= i:
            return None
        self.stats[i].add(x[i :: self.b])
        return self.stats[i].buffer

    def _calc_prev_buffer(self, i: int, x: np.ndarray) -> Optional[np.ndarray]:
        """Calculate the updated buffer for the i-th batch.

        Parameters
        ----------
        i : int >= 0
            The index of the batch.

        x : np.ndarray [shape=(n, d)]
            The data points.

        Returns
        -------
        out : np.ndarray [shape=(p + 1, d)]
            The updated buffer.

        """
        if len(x) <= i:
            return None
        self.stats[i].sub(x[i :: self.b])
        return self.stats[i].buffer
