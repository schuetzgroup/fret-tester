# Copyright 2017 Lukas Schrangl
"""Fluorophore brightness distribution models"""
import numbers
import numpy as np


class LognormalBrightness:
    """Fluorophore brightness modeled by a lognormal distribution"""
    def __init__(self, max_mean=None, precision=0.):
        """Parameters
        ----------
        max_mean : float or None
            If given (and `precision` > 0), precalculate parameters of the
            lognormal distributions (corresponding to means and matchings stds)
            from 0 to `max_mean` in steps of `precision` for a performance
            gain (at cost of accuracy). Defaults to None, i.e. no
            pre-computation.
        precision : float or None
            Precision of pre-computed lognormal parameters. The smaller, the
            more accurate, as long as it is > 0. Defaults to 0, i.e. no
            pre-computation.
        """
        self._cached_params = None
        self._cached_precision = precision
        self._test = 0

        if (isinstance(max_mean, numbers.Number) and max_mean > 0. and
                precision > 0.):
            # Pre-compute lognormal distribution parameters to save time later
            # (calculating log is expensive)
            p = self._parameters(np.arange(0, max_mean+precision, precision))
            self._cached_params = np.array(p).T

    def _parameters(self, m):
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
        s = self.std_from_mean(m)

        mu = np.empty_like(m, dtype=float)
        sigma = np.empty_like(s, dtype=float)

        # m > 0; m == 0 needs special treatment since log() will fail
        gt0 = m > 0
        m_gt0 = m[gt0]
        s_gt0 = s[gt0]

        x = 1 + s_gt0**2 / m_gt0**2
        mu[gt0] = np.log(m_gt0 / np.sqrt(x))  # Inverse formulas for lognormal
        sigma[gt0] = np.sqrt(np.log(x))       # mean and std

        # m == 0 ==> mu = -np.inf
        mu[~gt0] = -np.inf
        sigma[~gt0] = 0

        return mu, sigma

    def generate(self, m):
        """Draw random value from the brightness distribution with given mean

        Parameters
        ----------
        m : array_like
            Mean values. For each entry, draw a random number from the
            brightness distribution whose mean value is this entry.

        Returns
        -------
        numpy.ndarray
            Random brightness values
        """
        if self._cached_params is None:
            mu, sigma = self._parameters(m)
        else:
            idx = np.round(m / self._cached_precision).astype(int)
            mu, sigma = self._cached_params[idx].T

        if self._test:
            return mu * sigma
        else:
            return np.random.lognormal(mu, sigma)

    def __call__(self, m):
        return self.generate(m)
