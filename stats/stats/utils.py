# Copyright (c) 2024 Takenori Yoshimura
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np


def power(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Computes the power of x1 to x2.

    Parameters
    ----------
    x1 : np.ndarray [shape=(..., d)]
        The base.

    x2 : np.ndarray [shape=(p + 1,)]
        The exponents.

    Returns
    -------
    out : np.ndarray [shape=(..., p + 1, d)]
        The power of x1 to x2.

    """
    return np.power(np.expand_dims(x1, -2), np.expand_dims(x2, -1))
