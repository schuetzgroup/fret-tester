import unittest
import os

import numpy as np

from fret_simulator import time_trace


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_image")


def fake_rand():
    return 0


def fake_rand_exp(m, shape):
    return np.broadcast_to(m, shape)


class TestTwoStateExpTruth(unittest.TestCase):
    """Test the `TwoStateExpTruth` class"""
    def setUp(self):
        # Remove randomness from the output
        time_trace._rand = fake_rand  # always start in state 1
        time_trace._rand_exp = fake_rand_exp

        self.lifetimes = np.array([2., 4.])
        self.eff = np.array([0.8, 0.2])

    def test_call(self):
        """time_trace.TwoStateExpTruth.__call__: Basic functionality"""
        truth = time_trace.TwoStateExpTruth(self.lifetimes, self.eff)

        num_cycles = 10
        dur = num_cycles * self.lifetimes.sum()
        t, e = truth(dur-1)

        np.testing.assert_allclose(
            t[:2*num_cycles],
            np.cumsum(self.lifetimes.tolist()*num_cycles))
        np.testing.assert_allclose(e[:2*num_cycles],
                                   self.eff.tolist()*num_cycles)

    def test_call_loop(self):
        """time_trace.TwoStateExpTruth.__call__: Trigger the loop

        Multiple concatenated simulation runs are needed to generate a trace
        that is long enough by letting the fake random number generator return
        very small numbers for lifetimes.
        """
        lt_scale = 10
        time_trace._rand_exp = lambda m, s: fake_rand_exp(m, s) / lt_scale

        truth = time_trace.TwoStateExpTruth(self.lifetimes, self.eff)

        dur_mult = 10
        dur = self.lifetimes.sum() * dur_mult
        num_cycles = dur_mult * lt_scale
        t, e = truth(dur-1/lt_scale)

        np.testing.assert_allclose(
            t[:2*num_cycles],
            np.cumsum((self.lifetimes/lt_scale).tolist()*num_cycles))
        np.testing.assert_allclose(e[:2*num_cycles],
                                   self.eff.tolist()*num_cycles)

    def test_call_short(self):
        """time_trace.TwoStateExpTruth.__call__: Minimum length

        Make sure at least one data point is generated.
        """
        truth = time_trace.TwoStateExpTruth(self.lifetimes, self.eff)

        t, e = truth(1)
        assert(len(t) >= 1)
        assert(len(e) >= 1)


class TestMakeStepFunction(unittest.TestCase):
    """Test the `make_step_function` function"""
    def test_call(self):
        """time_trace.make_step_function: Basic functionality"""
        x = [1, 2, 3, 4]
        y = [0, 1, 0, 1]

        x2, y2 = time_trace.make_step_function(x, y)

        np.testing.assert_allclose(x2, [0, 1, 1, 2, 2, 3, 3, 4])
        np.testing.assert_allclose(y2, [0, 0, 1, 1, 0, 0, 1, 1])


class TestSample(unittest.TestCase):
    """Test the `sample` function"""
    def test_call(self):
        """time_trace.sample: Basic functionality"""
        t = np.arange(3, 37, 3)
        eff = [0.8, 0.2]
        e = np.array(eff*6)

        s = time_trace.sample(t, e, 4)

        f = np.array([0.75, 0.5, 0.25])
        f = np.column_stack((f, f[::-1]))
        se_exp = np.sum(f * np.array([eff]), axis=1)

        np.testing.assert_allclose(
            s, [np.arange(4, 37, 4), se_exp.tolist()*3])

    def test_long_trace(self):
        """time_trace.sample: Check for rounding errors in long traces

        Make sure that integration of the whol trace and using np.diff on it
        does not lead to cancellation effects.
        """
        t = np.linspace(0.5, 5000000.5, 5000000, endpoint=False)
        eff = [0.8, 0.2]
        e = np.tile(eff, 2500000)
        print(e[:10])

        s = time_trace.sample(t, e, 1)
        np.testing.assert_allclose(s[1], np.full(4999999, 0.5),
                                   atol=0., rtol=1e-10)


class TestExperiment(unittest.TestCase):
    """Test the `experiment` function"""
    def test_call(self):
        """time_trace.experiment: Basic functionality"""
        e = np.tile([0.8, 0.2], 10)
        phot = 100

        def donor(m):
            return 3 * m

        def acceptor(m):
            return 2 * m

        d, a = time_trace.experiment(e, 100, donor, acceptor)

        np.testing.assert_allclose(d, 3*(1-e)*phot)
        np.testing.assert_allclose(a, 2*e*phot)


class TestSimulateDataset(unittest.TestCase):
    """Test the `simulate_dataset` function"""
    def setUp(self):
        # Remove randomness from the output
        time_trace._rand = fake_rand  # always start in state 1
        time_trace._rand_exp = fake_rand_exp

        self.lifetimes = np.array([2., 4.])
        self.eff = np.array([0.8, 0.2])

    def test_call(self):
        """time_trace.simulate_dataset: Basic functionality"""
        truth = time_trace.TwoStateExpTruth(self.lifetimes, self.eff)
        t_ex = 3
        dp = 10000
        phot = 100

        def donor(m):
            return 3 * m

        def acceptor(m):
            return 2 * m

        d = time_trace.simulate_dataset(truth, t_ex, dp, phot, donor,
                                        acceptor)

        np.testing.assert_allclose([d.true_time, d.true_eff],
                                   truth(dp*t_ex))
        np.testing.assert_allclose(
            [d.samp_time, d.samp_eff],
            time_trace.sample(d.true_time, d.true_eff, t_ex))

        db, ab = time_trace.experiment(d.samp_eff, phot, donor, acceptor)
        np.testing.assert_allclose(d.exp_don, db)
        np.testing.assert_allclose(d.exp_acc, ab)
        np.testing.assert_allclose(d.exp_eff, ab/(db+ab))

    def test_truth_array(self):
        """time_trace.simulate_dataset: Pass array as truth parameter"""
        truth = time_trace.TwoStateExpTruth(self.lifetimes, self.eff)
        t_ex = 3
        dp = 10000
        phot = 100

        def donor(m):
            return 3 * m

        def acceptor(m):
            return 2 * m

        truth_array = truth(dp*t_ex)
        d = time_trace.simulate_dataset(truth_array, t_ex, dp, phot, donor,
                                        acceptor)

        np.testing.assert_allclose([d.true_time, d.true_eff],
                                   truth_array)
        np.testing.assert_allclose(
            [d.samp_time, d.samp_eff],
            time_trace.sample(d.true_time, d.true_eff, t_ex))

        db, ab = time_trace.experiment(d.samp_eff, phot, donor, acceptor)
        np.testing.assert_allclose(d.exp_don, db)
        np.testing.assert_allclose(d.exp_acc, ab)
        np.testing.assert_allclose(d.exp_eff, ab/(db+ab))


if __name__ == "__main__":
    unittest.main()