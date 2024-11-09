# Copyright (c) 2024 Takenori Yoshimura
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseStats(ABC):
    @abstractmethod
    def get(self) -> np.ndarray:
        """Return the moments of the data.

        Returns
        -------
        out : np.ndarray [shape=(p + 1, d)]
            The moments of the data.

        """
        raise NotImplementedError

    @abstractmethod
    def add(self, x: np.ndarray) -> None:
        """Add a new data point(s) to the statistics.

        Parameters
        ----------
        x : np.ndarray [shape=(d,) or (n, d)]
            The data point(s) to add.

        """
        raise NotImplementedError

    @abstractmethod
    def merge(self, stats: BaseStats) -> None:
        """Merge the statistics with another set of statistics.

        Parameters
        ----------
        stats : BaseStats
            The statistics to merge with.

        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear the accumulated statistics."""
        raise NotImplementedError
