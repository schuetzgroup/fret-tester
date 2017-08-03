# Copyright 2017 Lukas Schrangl
"""Tools for simulating two-state smFRET time traces"""
from collections import namedtuple

import numpy as np
import scipy


def two_state_truth(life_times, efficiencies, duration):
    """Simulate ground truth smFRET time trace for a two state system

    Assume that the states' life times are exponentially distributed.

    Parameters
    ----------
    life_times : array_like, shape=(2,)
        Mean (of the exponentially distributed) life times of the states
    efficiencies : array_like, shape=(2,)
        FRET efficiencies for the states
    duration : float
        Minimum duration of the time trace. In practice, it will be a little
        longer unless a state change occurs incidentally exactly at the
        specified time, for which the probability is 0.

    Returns
    -------
    time : numpy.ndarray
        Time points of state changes. These are bi-exponentially distributed.
    eff : numpy.ndarray
        Corresponding FRET efficiencies. The `i`-th entry lasts from
        ``time[i-1]`` (or 0 for i=0) to ``time[i]``.

    See also
    --------
    make_step_function : Create step function from data returned by this
    """
    num_states = len(life_times)
    # num_events increased by 50% to be (quite) sure that we generate
    # a long enough trace in one run in the `while` loop below
    num_events = int(duration / np.mean(life_times) * 1.5) // num_states

    prob_1 = life_times[0] / np.sum(life_times)  # prob. to start with state 1
    start_with_1 = np.random.rand() < prob_1  # whether to start with state 1

    time = []
    eff = []
    dur = 0
    # Run in a loop for the very unlikely case where not enough transitions
    # were generated (i.e. all randomly generated life times were very short)
    while dur < duration:
        # Generate transition times by generating exponentially distributed
        # random numbers
        t = np.random.exponential(life_times, (num_events, num_states))
        # FRET efficiencies are constant, just broadcast the array
        e = np.broadcast_to(efficiencies, (num_events, num_states))
        if not start_with_1:
            # Switch columns so that state 2 is first when flatten()ing below
            t = t[:, ::-1]
            e = e[:, ::-1]

        # Generate time series by merging the transition times
        t = np.cumsum(t.flatten())
        if time:
            t += time[-1][-1]
        time.append(t)
        dur = t[-1]

        # Generate FRET efficiency sequence by merging
        eff.append(e.flatten())

    time = np.concatenate(time)
    eff = np.concatenate(eff)

    # Make result as short as possible while still being longer than `duration`
    long_idx = np.nonzero(time > duration)[0][0] + 1
    return time[:long_idx], eff[:long_idx]


def make_step_function(x, y):
    """Turn data from :py:func:`two_state_truth` into a step function

    The returned data can e.g. easily be plotted. The function works by
    duplicating each point in the `x` and `y` arrays and shifting them by one
    index so that for each `x` entry one has both parts of the step in `y`.

    If ``x = [1, 2, 3, 4]`` and ``y = [0, 1, 0, 1]`` (meaning the data
    jump from 0 to 1 at x=1, from 1 to 0 at x=2, …), a call to this functions
    will return ``x = [0, 1, 1, 2, 2, 3, 3, 4]`` and
    ``y = [0, 0, 1, 1, 0, 0, 1, 1]``.

    Parameters
    ----------
    x, y : array_like
        x and y axis values

    Returns
    -------
    x, y : numpy.ndarray
        x and y axis values for the step function.
    """
    x2 = np.roll(np.repeat(x, 2), 1)
    x2[0] = 0
    return x2, np.repeat(y, 2)


def sample(time, eff, exposure_time, data_points=np.inf):
    """Sample the true smFRET time trace with finite exposure time

    This means that there will be integration over all transitions that happen
    during a single exposure.

    Parameters
    ----------
    time : array_like, shape=(n,)
        Sequence of transition times as returned by :py:func:`two_state_truth`
    eff : array_like, shape=(n,)
        Sequence of FRET efficiencies as returned by :py:func:`two_state_truth`
    exposure_time : float
        Exposure time for sampling. All transition during a exposure will be
        integrated.
    data_points : scalar, optional
        Number of data points to return. If the FRET time trace is too short,
        the maximum number of data points the trace allows is returned.
        Defaults to infinity.

    Returns
    -------
    sample_time : numpy.ndarray
        Time points of sampling. They are evenly spaced with `exposure_time`
        difference.
    sample_eff : numpy.ndarray
        Corresponding sampled FRET efficiencies
    """
    step_t, step_eff = make_step_function(time, eff)
    data_points = int(min(time[-1] / exposure_time, data_points))

    sample_t = np.linspace(0, data_points*exposure_time, data_points+1,
                           endpoint=True)

    # Integrate the step function
    int_eff = scipy.integrate.cumtrapz(step_eff, step_t, initial=0)
    # Get integrated function at points of interest (i.e. sampling time points)
    sample_int_eff = np.interp(sample_t, step_t, int_eff,
                               left=np.NaN, right=np.NaN)
    # Now take derivative which gives the integrated efficiencies between the
    # sampling time points
    sample_eff = np.diff(sample_int_eff)
    # Scale correctly
    sample_eff /= exposure_time

    return sample_t[1:], sample_eff


def experiment(eff, photons, donor_brightness, acceptor_brightness):
    """From sampled smFRET time traces, simulate (noisy) experiment results

    This adds noise coming from the rather broad brightness distributions of
    fluorophores to the time traces, resulting in a trace as one would get
    out of an experiment.

    Parameters
    ----------
    eff : array_like
        Sampled FRET efficiencies
    photons : float
        Mean summed number of photons emitted by donor and acceptor per
        exposure.
    donor_brightness, acceptor_brightness : callable
        Takes one argument, an array of mean brightness values. For each entry
        `m` it returns a random value drawn from the brightness distribution
        with mean `m`.
    """
    acc_p = eff * photons
    don_p = (1 - eff) * photons
    acc_p_noisy = acceptor_brightness(acc_p)
    don_p_noisy = donor_brightness(don_p)
    return don_p_noisy, acc_p_noisy


DataSet = namedtuple("DataSet", ["true_time", "true_eff",
                                 "samp_time", "samp_eff",
                                 "exp_don", "exp_acc", "exp_eff"])
DataSet.__doc__ = """Named tuple containing a full data set of a simulation run

Attributes
----------
true_time, true_eff : array_like
    True state trajectory as produced by :py:func:`two_state_truth`
samp_time, samp_eff : array_like
    Sampled state trajectory as produced by :py:func:`sample`
exp_don, exp_acc : array_like
    (Noisy) donor and acceptor intensities as produced by
    :py:func:`experiment`
exp_eff : array_like
    Experiment FRET efficiency, i.e. ``exp_acc / (exp_acc + exp_don)``
"""


def simulate_dataset(life_times, efficiencies, exposure_time, data_points,
                     photons, donor_brightness, acceptor_brightness,
                     truth=None):
    """Simulate a whole data set

    Consecutively run :py:func:`two_state_truth`, :py:func:`sample`, and
    :py:func:`experiment`.

    Parameters
    ----------
    life_times : array_like, shape=(2,)
        Mean (of the exponentially distributed) life times of the states
    efficiencies : array_like, shape=(2,)
        FRET efficiencies for the states
    exposure_time : float
        Exposure time for sampling. All transition during a exposure will be
        integrated.
    data_points : scalar, optional
        Number of data points to return. If the FRET time trace is too short,
        the maximum number of data points the trace allows is returned.
        Defaults to infinity.
    photons : float
        Mean summed number of photons emitted by donor and acceptor per
        exposure.
    donor_brightness, acceptor_brightness : callable
        Takes one argument, an array of mean brightness values. For each entry
        `m` it returns a random value drawn from the brightness distribution
    truth : tuple of array_like or None, optional
        It is possible to pass the result of a :py:func:`two_state_truth`
        call here. In this case, no new truth will be constructed (and
        thus the `lifetimes` parameters are ignored), but this will be used.
        Defaults to `None`.

    Returns
    -------
    DataSet
        Collected simulated data
    """
    dur = data_points * exposure_time
    if truth is None:
        t, e = two_state_truth(life_times, efficiencies, dur)
    else:
        t, e = truth
    st, se = sample(t, e, exposure_time, data_points)
    d, a = experiment(se, photons, donor_brightness, acceptor_brightness)
    return DataSet(t, e, st, se, d, a, a/(d+a))
