import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def gcd(a, b):
    while b > 0:
        t = b
        b = a % b
        a = t
    return a


@numba.jit(nopython=True, cache=True)
def lcm(a, b):
    return a * b / gcd(a, b)


@numba.jit(nopython=True, cache=True)
def ks_2samp(data1, data2):
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
