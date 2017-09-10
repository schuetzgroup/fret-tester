# Copyright 2017 Lukas Schrangl
"""Fluorophore brightness distribution models (numba JIT accelerated)"""
import numpy as np
import numba


class LognormalBrightness:
    def __init__(self, max_mean, precision):
        self._cached_precision = precision
        if max_mean > 0:
            self._cached_params = self._parameters(
                np.arange(0, max_mean+precision, precision)).T
        else:
            self._cached_params = np.empty((2, 0))

        self._test = 0

    def _parameters_scalar(self, m):
        """Calculate lognormal parameters from desired mean

        Compute :math:`\mu` and :math:`\sigma` of the Gaussian distribution
        that the lognormal distribution is based on such that the lognormal
        distribution will have mean `m` and standard deviation
        ``std_from_mean(m)`` (see :py:meth:`__init__`).

        Parameters
        ----------
        m : array_like
            Mean value for which the lognormal parameters are desired

        Returns
        -------
        mu : numpy.ndarray
            :math:`\mu` of the underlying Gaussian
        sigma : numpy.ndarray
            :math:`\sigma` of the underlying Gaussian
        """
        if m > 0:
            s = self.std_from_mean(m)
            x = 1 + s**2 / m**2
            mu = np.log(m / np.sqrt(x))
            sigma = np.sqrt(np.log(x))
        else:
            mu = -np.inf
            sigma = 0.

        return mu, sigma

    def _parameters(self, m):
        ret = np.empty((2, m.size))
        for i in range(m.size):
            ret[0, i], ret[1, i] = self._parameters_scalar(m[i])
        return ret

    def generate(self, m):
        ret = np.empty(m.size)
        if self._cached_params.size > 0:
            for i in range(m.size):
                idx = int(np.round(m[i] / self._cached_precision))
                mu = self._cached_params[idx, 0]
                sigma = self._cached_params[idx, 1]
                if not self._test:
                    ret[i] = np.random.lognormal(mu, sigma)
                else:
                    ret[i] = mu * sigma
        else:
            for i in range(m.size):
                mu, sigma = self._parameters_scalar(m[i])
                if not self._test:
                    ret[i] = np.random.lognormal(mu, sigma)
                else:
                    ret[i] = mu * sigma
        return ret


def lognormal_jitclass(class_or_spec):
    lnspec = [("_cached_precision", numba.float64),
              ("_cached_params", numba.float64[:, :]),
              ("_test", numba.int64)]
    if isinstance(class_or_spec, type):
        return numba.jitclass(lnspec)(class_or_spec)
    else:
        return numba.jitclass(class_or_spec + lnspec)
