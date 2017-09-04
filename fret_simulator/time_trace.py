# Copyright 2017 Lukas Schrangl
"""Tools for simulating smFRET time traces"""
from collections import namedtuple

import numpy as np


# Use these variables to be able to replace random number generators by
# something deterministic for testing
_rand = np.random.rand
_rand_exp = np.random.exponential


class TwoStateExpTruth:
    """Simulate ground truth smFRET time traces (two states, exp. lifetimes)

    Lifetimes are exponentially distributed.
    """
    def __init__(self, lifetimes, efficiencies):
        """Parameters
        ----------
        life_times : array_like, shape=(2,)
            Mean (of the exponentially distributed) life times of the states
        efficiencies : array_like, shape=(2,)
            FRET efficiencies for the states
        """
        self.lifetimes = lifetimes
        self.efficiencies = efficiencies

    def __call__(self, duration):
        """Create a time trace that of at least `duration` length

        Parameters
        ----------
        duration : float
            Minimum duration of the time trace. In practice, it will be
            longer.

        Returns
        -------
        time : numpy.ndarray
            Time points of state changes. These are bi-exponentially
            distributed.
        eff : numpy.ndarray
            Corresponding FRET efficiencies. The `i`-th entry lasts from
            ``time[i-1]`` (or 0 for i=0) to ``time[i]``.

        See also
        --------
        make_step_function : Create step function from data returned by this
        """
        num_states = len(self.lifetimes)
        # num_events increased by 50% to be (quite) sure that we generate
        # a long enough trace in one run in the `while` loop below
        num_events = (int(duration / np.mean(self.lifetimes) * 1.5) //
                      num_states) or 1  # minimum is one event

        # prob. to start in state 1
        prob_1 = self.lifetimes[0] / np.sum(self.lifetimes)
        start_with_1 = _rand() < prob_1  # whether to start with state 1

        time = []
        eff = []
        dur = 0
        # Run in a loop for the very unlikely case where not enough transitions
        # were generated (i.e. all randomly generated life times were very
        # short)
        while dur < duration:
            # Generate transition times by generating exponentially distributed
            # random numbers
            t = _rand_exp(self.lifetimes, (num_events, num_states))
            # FRET efficiencies are constant, just broadcast the array
            e = np.broadcast_to(self.efficiencies, (num_events, num_states))
            if not start_with_1:
                # Switch columns so that state 2 is first when flatten()ing
                # below
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

        # Make result as short as possible while still being longer than
        # `duration`
        long_idx = np.nonzero(time > duration)[0][0] + 1
        return time[:long_idx], eff[:long_idx]


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
    # Prepend 0 to `time`
    t0 = np.empty(len(time)+1)
    t0[0] = 0
    t0[1:] = time

    data_points = int(min(time[-1] / exposure_time, data_points))

    sample_t = np.linspace(0, data_points*exposure_time, data_points+1,
                           endpoint=True)

    # Integrate the efficiency step function, prepend 0
    lifetimes = np.diff(t0)
    int_eff = np.empty(len(t0))
    int_eff[0] = 0
    np.cumsum(np.multiply(lifetimes, eff), out=int_eff[1:])
    # Get integrated function at points of interest (i.e. sampling time points)
    sample_int_eff = np.interp(sample_t, t0, int_eff,
                               left=np.NaN, right=np.NaN)
    # Now take use diff to get the integrated efficiencies between the
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


def simulate_dataset(truth, exposure_time, data_points, photons,
                     donor_brightness, acceptor_brightness):
    """Simulate a whole data set

    Consecutively run :py:func:`two_state_truth`, :py:func:`sample`, and
    :py:func:`experiment`.

    Parameters
    ----------
    truth : callable or tuple of numpy.ndarray, shape=(n,)
        Ground truth smFRET time trace. If this is an array, ``truth[0]``
        should be the sequence of time points where state transitions
        occur and ``truth[1]`` the FRET efficiency between the (i-1)-th
        time point (or 0 for i = 0) and the i-th time point.
        If this is callable, it should return a ground truth smFRET time
        trace as described above. It should take one float parameter, which is
        the minimum duration of the generated trace.
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
        `m` it returns a random value drawn from the brightness distribution.

    Returns
    -------
    DataSet
        Collected simulated data
    """
    dur = data_points * exposure_time
    if callable(truth):
        t, e = truth(dur)
    else:
        t, e = truth

    st, se = sample(t, e, exposure_time, data_points)
    d, a = experiment(se, photons, donor_brightness, acceptor_brightness)
    return DataSet(t, e, st, se, d, a, a/(d+a))
