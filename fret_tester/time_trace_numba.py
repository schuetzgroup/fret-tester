import numba
import numpy as np


_two_state_exp_truth_spec = [("lifetimes", numba.float64[:]),
                             ("efficiencies", numba.float64[:]),
                             ("_test", numba.int64)]


@numba.jitclass(_two_state_exp_truth_spec)
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

        See also
        --------
        make_step_function : Create step function from data returned by this
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
                return ret_t, ret_e

        # Did not produce a long enough time trace
        # Fall back to slower dynamically growing list
        extra_t = []
        extra_e = []
        while t < duration:
            t += self._rand_exp(self.lifetimes[int(s)])
            extra_t.append(t)
            extra_e.append(self.efficiencies[int(s)])
            s = not s
            i += 1

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
