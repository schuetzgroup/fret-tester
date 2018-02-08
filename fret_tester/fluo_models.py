"""Fluorophore brightness distribution models"""
# Copyright 2017-2018 Lukas Schrangl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numbers
import numpy as np


class LognormalBrightness:
    """Fluorophore brightness modeled by a lognormal distribution

    This is intended as a base class. One needs to implement the
    :py:meth:`std_from_mean` in a subclass which gives the lognormal
    distribution standard deviation for a certain mean.
    """
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
            p = self.parameters(np.arange(0, max_mean+precision, precision))
            self._cached_params = np.array(p).T

    def parameters(self, m):
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
            mu, sigma = self.parameters(m)
        else:
            idx = np.round(m / self._cached_precision).astype(int)
            mu, sigma = self._cached_params[idx].T

        if self._test:
            return mu * sigma
        else:
            return np.random.lognormal(mu, sigma)

    def __call__(self, m):
        """Synonymous for :py:meth:`generate`"""
        return self.generate(m)

    def std_from_mean(self, m):
        """Get standard deviation for given mean

        This needs to be implemented in a subclass.

        Parameters
        ----------
        m : array-like
            Mean values to calculate standard deviations for.

        Returns
        -------
        numpy.ndarray
            Standard deviations corresponding to means.
        """
        raise NotImplementedError("`std_from_mean` needs to be implemented.")


class PolyLnBrightness(LognormalBrightness):
    """LognormalBrightness subclass with polynomial mean-vs.-std relation

    This is a subclass of :py:class:`LognormalBrightness` where the relation
    between mean and standard deviation is described by a polynomial.
    """
    def __init__(self, parms, max_mean=None, precision=0.):
        """Parameters
        ----------
        parms : list of float
            Polynomial coefficients in decreasing order
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
        self.poly = np.poly1d(parms)
        super().__init__(max_mean, precision)

    def std_from_mean(self, m):
        """Get std. deviation for given mean according to polynomial relation

        Parameters
        ----------
        m : array-like
            Mean values to calculate standard deviations for.

        Returns
        -------
        numpy.ndarray
            Standard deviations corresponding to means.
        """
        return self.poly(m)
