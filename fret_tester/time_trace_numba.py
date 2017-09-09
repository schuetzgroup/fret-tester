# Copyright 2017 Lukas Schrangl
"""Tools for simulating smFRET time traces (numba JIT accelerated)"""
import numba
import numpy as np


@numba.jitclass([("lifetimes", numba.float64[:]),
                 ("efficiencies", numba.float64[:]),
                 ("_test", numba.int64)])
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
        # Copy lifetimes and efficiencies. This way, an array can be passed
        # as well as a list
        self.lifetimes = np.empty(len(lifetimes), dtype=np.float64)
        self.efficiencies = np.empty(len(efficiencies), dtype=np.float64)
        for i in range(len(lifetimes)):
            self.lifetimes[i] = float(lifetimes[i])
            self.efficiencies[i] = float(efficiencies[i])

        # If 1 or 2, disable random number generation and yield predictable
        # results. See _rand_uniform and _rand_exp methods
        self._test = 0

    def generate(self, duration):
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
        """
        # num_events increased by 50% to be (quite) sure that we generate
        # a long enough trace in one run in the `while` loop below
        # minimum is one event
        num_events = max(int(1.5 * duration / np.mean(self.lifetimes)), 1)

        # prob. to start in state 1
        prob_1 = self.lifetimes[0] / np.sum(self.lifetimes)
        s = self._rand_uniform() >= prob_1  # If False, start with state 1

        ret_t = np.full(num_events, np.NaN)
        ret_e = np.full(num_events, np.NaN)
        t = 0
        i = 0
        while i < num_events:
            t += self._rand_exp(self.lifetimes[int(s)])
            ret_t[i] = t
            ret_e[i] = self.efficiencies[int(s)]
            s = not s
            i += 1
            if t > duration:
                return ret_t[:i], ret_e[:i]

        # Did not produce a long enough time trace
        # Fall back to slower dynamically growing list
        extra_t = []
        extra_e = []
        while t < duration:
            t += self._rand_exp(self.lifetimes[int(s)])
            extra_t.append(t)
            extra_e.append(self.efficiencies[int(s)])
            s = not s

        ret_t2 = np.empty(len(ret_t) + len(extra_t))
        ret_e2 = np.empty(len(ret_t) + len(extra_t))
        for i in range(len(ret_t)):
            ret_t2[i] = ret_t[i]
            ret_e2[i] = ret_e[i]
        offset = len(ret_t)
        for i in range(len(extra_t)):
            ret_t2[offset+i] = extra_t[i]
            ret_e2[offset+i] = extra_e[i]

        return ret_t2, ret_e2

    def _rand_exp(self, m):
        if self._test == 0:
            return np.random.exponential(m)
        if self._test == 1:
            return m
        if self._test == 2:
            return m / 10
        return np.random.exponential(m)

    def _rand_uniform(self):
        if self._test == 0:
            return np.random.rand()
        if (self._test == 1) or (self._test == 2):
            return 0
        return np.random.rand()


@numba.njit
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
    num_dp = int(min(time[-1] / exposure_time, data_points))

    ret_t = np.empty(num_dp)
    ret_e = np.empty(num_dp)

    t_idx = 0
    for dp in range(0, num_dp):
        t_end = (dp + 1) * exposure_time
        ret_t[dp] = t_end

        intens = 0.
        t_sub = t_end - exposure_time
        while time[t_idx] < t_end:
            intens += (time[t_idx] - t_sub) * eff[t_idx]
            t_sub = time[t_idx]
            t_idx += 1
        intens += (t_end - t_sub) * eff[t_idx]

        ret_e[dp] = intens / exposure_time
    return ret_t, ret_e


@numba.njit
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
    acc_p_noisy = acceptor_brightness.generate(acc_p)
    don_p_noisy = donor_brightness.generate(don_p)
    return don_p_noisy, acc_p_noisy


@numba.jitclass([("true_time", numba.float64[:]),
                 ("true_eff", numba.float64[:]),
                 ("samp_time", numba.float64[:]),
                 ("samp_eff", numba.float64[:]),
                 ("exp_don", numba.float64[:]),
                 ("exp_acc", numba.float64[:]),
                 ("exp_eff", numba.float64[:])])
class DataSet:
    """Full data set of a simulation run

    Attributes
    ----------
    true_time, true_eff : numpy.ndarray, dtype(float64)
        True state trajectory as produced by :py:func:`two_state_truth`
    samp_time, samp_eff : numpy.ndarray, dtype(float64)
        Sampled state trajectory as produced by :py:func:`sample`
    exp_don, exp_acc : numpy.ndarray, dtype(float64)
        (Noisy) donor and acceptor intensities as produced by
        :py:func:`experiment`
    exp_eff : numpy.ndarray, dtype(float64)
        Experiment FRET efficiency, i.e. ``exp_acc / (exp_acc + exp_don)``
"""
    def __init__(self, true_time, true_eff, samp_time, samp_eff, exp_don,
                 exp_acc, exp_eff):
        self.true_time = true_time
        self.true_eff = true_eff
        self.samp_time = samp_time
        self.samp_eff = samp_eff
        self.exp_don = exp_don
        self.exp_acc = exp_acc
        self.exp_eff = exp_eff


@numba.njit
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
    t, e = truth.generate(dur)

    st, se = sample(t, e, exposure_time, data_points)
    d, a = experiment(se, photons, donor_brightness, acceptor_brightness)
    return DataSet(t, e, st, se, d, a, a/(d+a))
