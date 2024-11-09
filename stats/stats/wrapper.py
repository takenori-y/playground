# Copyright (c) 2024 Takenori Yoshimura
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Any

import numpy as np

from .base import BaseStats


class StatsWrapper(BaseStats):
    def __init__(self, p: int, mode: str = "naive", **kwargs: Any) -> None:
        """Initialize the class.

        Parameters
        ----------
        p : int >= 0
            The maximum order of the moments to be calculated.

        mode : ["naive", "online", "batch"]
            The mode to use for calculating the statistics.

        **kwargs
            Additional keyword arguments.

        """
        stats: BaseStats
        if mode == "naive":
            from .naive import NaiveStats

            stats = NaiveStats(p, **kwargs)
        elif mode == "online":
            from .online import OnlineStats

            stats = OnlineStats(p, **kwargs)
        elif mode == "batch":
            from .batch import BatchedOnlineStats

            stats = BatchedOnlineStats(p, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.stats = stats

    def get(self) -> np.ndarray:
        """Return the moments of the data.

        Returns
        -------
        out : np.ndarray [shape=(p + 1, d)]
            The moments of the data.

        """
        return self.stats.get()

    def add(self, x: np.ndarray) -> None:
        """Add a new data point(s) to the statistics.

        Parameters
        ----------
        x : np.ndarray [shape=(d,) or (n, d)]
            The data point(s) to add.

        """
        self.stats.add(x)

    def sub(self, x: np.ndarray) -> None:
        """Subtract a data point(s) from the statistics.

        Parameters
        ----------
        x : np.ndarray [shape=(d,) or (n, d)]
            The data point(s) to subtract.

        """
        self.stats.sub(x)

    def merge(self, stats: BaseStats) -> None:
        """Merge the statistics with another set of statistics.

        Parameters
        ----------
        stats : BaseStats
            The statistics to merge with.

        """
        if not isinstance(stats, type(self)):
            raise ValueError("The input must be an object of the same class.")
        self.stats.merge(stats.stats)

    def purge(self, stats: BaseStats) -> None:
        """Purge the statistics with another set of statistics.

        Parameters
        ----------
        stats : BaseStats
            The statistics to purge with.

        """
        if not isinstance(stats, type(self)):
            raise ValueError("The input must be an object of the same class.")
        self.stats.purge(stats.stats)

    def clear(self) -> None:
        """Clear the accumulated statistics."""
        self.stats.clear()
