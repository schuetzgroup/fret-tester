# Copyright 2017 Lukas Schrangl
"""Some utility functions"""
import numpy as np
import scipy.stats


def _bi_beta_model(x, f, a1, b1, a2, b2):
    """Mixture of two Beta PDFs

    Returns f * beta.pdf(x, a1, b1) + (1 - f) * beta.pdf(x, a2, b2), where
    `beta.pdf` is from :py:module:`scipy.stats`.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa values
    f : float
        Weight of first beta distribution. Second distribution has weight
        1 - f.
    a1, b1 : float
        Shape parameters of first beta distribution
    a2, b2 : float
        Shape parameters of second beta distribution

    Returns
    -------
    numpy.ndarray
        PDF values at `x`.
    """
    b1 = scipy.stats.beta.pdf(x, a1, b1)
    b2 = scipy.stats.beta.pdf(x, a2, b2)
    return f * b1 + (1 - f) * b2


def bi_beta_fit(eff, bins=100):
    """Fit mixture of two Beta distribution PDFs to histogram

    Fits f * beta.pdf(x, a1, b1) + (1 - f) * beta.pdf(x, a2, b2) (where
    `beta.pdf` is from :py:module:`scipy.stats`) to the histogram of `f`
    using least squares fitting.

    Parameters
    ----------
    eff : numpy.ndarray
        Array of FRET efficiencies
    bins : int
        Number of bins for the histogram

    Returns
    -------
    f : float
        Weight of first beta distribution. Second distribution has weight
        1 - f.
    param1 : tuple of float
        Shape parameters for first Beta distribution, i.e. `(a1, b1)`.
    param2 : tuple of float
        Shape parameters for second Beta distribution, i.e. `(a2, b2)`.
    """
    h, bins = np.histogram(eff, bins=np.linspace(0, 1, bins+1), density=True)
    bin_ctr = (bins[:-1] + bins[1:]) / 2

    beta1_0 = [1., 2.]
    beta2_0 = [2., 1.]
    f_0 = [0.5]

    beta_bounds = ([1., 1.], [np.inf]*2)
    f_bounds = ([0.], [1.])

    fit, _ = scipy.optimize.curve_fit(
        _bi_beta_model,
        bin_ctr, h,
        p0=f_0+beta1_0+beta2_0,
        bounds=[fb+bb*2 for fb, bb in zip(f_bounds, beta_bounds)])

    return fit[0], fit[1:3], fit[3:5]
