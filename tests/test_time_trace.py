import unittest
import os

import numpy as np

from fret_tester import time_trace


path, f = os.path.split(os.path.abspath(__file__))


def make_two_state_exp_truth_seed0():
    """Generate "two_state_exp_truth_seed0.npz"

    Used for various tests.
    """
    np.random.seed(0)
    np.random.rand()  # Determine start state. Number is 0.55... -> state 2
    lt = np.random.exponential([4., 2.], (16, 2))  # gives a length of 120

    t = np.empty((lt.size))
    t[0::2] = lt[:, 0]
    t[1::2] = lt[:, 1]
    t = np.cumsum(t)

    e = np.empty_like(t)
    e[0::2] = 0.2
    e[1::2] = 0.8

    s = time_trace.sample(t, e, 3, 40)

    np.savez_compressed(os.path.join(path, "two_state_exp_truth_seed0.npz"),
                        t=t, e=e, s=s)


class TestTwoStateExpTruth(unittest.TestCase):
    """Test the `TwoStateExpTruth` class"""
    def setUp(self):
        self.lifetimes = np.array([2., 4.])
        self.eff = np.array([0.8, 0.2])
        self.truth_gen = time_trace.TwoStateExpTruth

    def test_generate(self):
        """time_trace.TwoStateExpTruth.generate: Basic functionality"""
        truth = self.truth_gen(self.lifetimes, self.eff)
        truth._test = 1

        num_cycles = 10
        dur = num_cycles * self.lifetimes.sum()
        t, e = truth.generate(dur-1)

        np.testing.assert_allclose(
            t[:2*num_cycles],
            np.cumsum(self.lifetimes.tolist()*num_cycles))
        np.testing.assert_allclose(e[:2*num_cycles],
                                   self.eff.tolist()*num_cycles)

    def test_generate_loop(self):
        """time_trace.TwoStateExpTruth.generate: Trigger the loop

        Multiple concatenated simulation runs are needed to generate a trace
        that is long enough by letting the fake random number generator return
        very small numbers for lifetimes.
        """
        lt_scale = 10
        truth = self.truth_gen(self.lifetimes, self.eff)
        truth._test = 2

        dur_mult = 10
        dur = self.lifetimes.sum() * dur_mult
        num_cycles = dur_mult * lt_scale
        t, e = truth.generate(dur-1/lt_scale)

        np.testing.assert_allclose(
            t[:2*num_cycles],
            np.cumsum((self.lifetimes/lt_scale).tolist()*num_cycles))
        np.testing.assert_allclose(e[:2*num_cycles],
                                   self.eff.tolist()*num_cycles)

    def test_generate_short(self):
        """time_trace.TwoStateExpTruth.generate: Minimum length

        Make sure at least one data point is generated.
        """
        truth = self.truth_gen(self.lifetimes, self.eff)
        truth._test = 1

        t, e = truth.generate(1)
        assert(len(t) >= 1)
        assert(len(e) >= 1)

    def test_generate_random_seed0(self):
        """time_trace.TwoStateExpTruth.generate: Use RNG with seed 0"""
        f = np.load(os.path.join(path, "two_state_exp_truth_seed0.npz"))
        t = f["t"]
        e = f["e"]

        truth = self.truth_gen(self.lifetimes, self.eff)
        np.random.seed(0)
        t_exp, e_exp = truth.generate(120)

        np.testing.assert_allclose(t_exp, t)
        np.testing.assert_allclose(e_exp, e)


class TestSample(unittest.TestCase):
    """Test the `sample` function"""
    def setUp(self):
        self.sample_func = time_trace.sample

    def test_call(self):
        """time_trace.sample: Basic functionality"""
        t = np.arange(3, 37, 3)
        eff = [0.8, 0.2]
        e = np.array(eff*6)

        s = self.sample_func(t, e, 4)

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

        s = self.sample_func(t, e, 1)
        np.testing.assert_allclose(s[1], np.full(4999999, 0.5),
                                   atol=0., rtol=1e-10)

    def test_random_input(self):
        """time_trace.sample: Use output of TwoStateExpTruth as input

        Compare to the saved result of a test run. The test run was not
        verified except using the other tests of this class, so this is more
        a regression test than a functionality test.
        """
        f = np.load(os.path.join(path, "two_state_exp_truth_seed0.npz"))
        s = self.sample_func(f["t"], f["e"], 3, 40)
        np.testing.assert_allclose(s, f["s"])


class TestExperiment(unittest.TestCase):
    """Test the `experiment` function"""
    def setUp(self):
        self.exp_func = time_trace.experiment
        self.donor_gen = lambda m: 3 * m
        self.acceptor_gen = lambda m: 2 * m

    def test_call(self):
        """time_trace.experiment: Basic functionality"""
        e = np.tile([0.8, 0.2], 10)
        phot = 100

        d, a = self.exp_func(e, 100, self.donor_gen, self.acceptor_gen)

        np.testing.assert_allclose(d, 3*(1-e)*phot)
        np.testing.assert_allclose(a, 2*e*phot)


class TestSimulateDataset(unittest.TestCase):
    """Test the `simulate_dataset` function"""
    def setUp(self):
        self.lifetimes = np.array([2., 4.])
        self.eff = np.array([0.8, 0.2])
        self.truth = time_trace.TwoStateExpTruth(self.lifetimes, self.eff)
        self.sample_func = time_trace.sample
        self.exp_func = time_trace.experiment
        self.dataset_func = time_trace.simulate_dataset
        self.donor_gen = lambda m: 3 * m
        self.acceptor_gen = lambda m: 2 * m

    def test_call(self):
        """time_trace.simulate_dataset: Basic functionality"""
        self.truth._test = 1
        t_ex = 3
        dp = 10000
        phot = 100

        d = self.dataset_func(self.truth, t_ex, dp, phot,
                              self.donor_gen, self.acceptor_gen)

        np.testing.assert_allclose([d.true_time, d.true_eff],
                                   self.truth.generate(dp*t_ex))
        np.testing.assert_allclose(
            [d.samp_time, d.samp_eff],
            self.sample_func(d.true_time, d.true_eff, t_ex))

        db, ab = self.exp_func(d.samp_eff, phot,
                               self.donor_gen, self.acceptor_gen)
        np.testing.assert_allclose(d.exp_don, db)
        np.testing.assert_allclose(d.exp_acc, ab)
        np.testing.assert_allclose(d.exp_eff, ab/(db+ab))

    def test_truth_array(self):
        """time_trace.simulate_dataset: Pass array as truth parameter"""
        self.truth._test = 1
        t_ex = 3
        dp = 10000
        phot = 100

        truth_array = self.truth.generate(dp*t_ex)
        d = self.dataset_func(truth_array, t_ex, dp, phot,
                              self.donor_gen, self.acceptor_gen)

        np.testing.assert_allclose([d.true_time, d.true_eff],
                                   truth_array)
        np.testing.assert_allclose(
            [d.samp_time, d.samp_eff],
            self.sample_func(d.true_time, d.true_eff, t_ex))

        db, ab = self.exp_func(d.samp_eff, phot,
                               self.donor_gen, self.acceptor_gen)
        np.testing.assert_allclose(d.exp_don, db)
        np.testing.assert_allclose(d.exp_acc, ab)
        np.testing.assert_allclose(d.exp_eff, ab/(db+ab))


if __name__ == "__main__":
    unittest.main()
