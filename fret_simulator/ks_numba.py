# Copyright 2017 Lukas Schrangl
"""Numba-accelerated two sample KS test and helper functions"""
import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def gcd(a, b):
    """Greatest common divisor

    Implementation of the Euclidean algorithm
    """
    while b > 0:
        t = b
        b = a % b
        a = t
    return a


@numba.jit(nopython=True, cache=True)
def lcm(a, b):
    """Least common multiple"""
    return a * b / gcd(a, b)


@numba.jit(nopython=True, cache=True)
def ks_2samp(data1, data2):
    """Calculate two-sample Kolmogorov-Smirnov (KS) statistic

    In order to compute the p-value from the statistic, use

    ::

        f = n1 * n2 / (n1 + n2)
        p_val = scipy.special.kolmogorov(np.sqrt(f) * ks)

    where ``n1`` and ``n2`` are the number of data points in each sample.

    Implementation of the 1D two-sample KS test described in [Xiao2017]_.

    .. [Xiao2017] Xiao, Y. A: "fast algorithm for two-dimensional
        Kolmogorov--Smirnov two sample tests", Computational Statistics & Data
        Analysis, Elsevier BV, 2017, 105, 53-58
    """
    n1 = len(data1)
    n2 = len(data2)
    n = n1 + n2
    L = lcm(n1, n2)
    d1 = L / n1
    d2 = L / n2

    data_all = np.concatenate((data1, data2))
    idx = data_all.argsort()

    h = np.empty(n)
    last_h = 0
    for i in range(n):
        last_h = last_h + d1 if idx[i] < n1 else last_h - d2
        h[i] = np.abs(last_h)
    return float(h.max()) / L
