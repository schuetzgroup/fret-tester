import unittest
import os

import numba
import numpy as np

from fret_tester import time_trace_numba
from . import test_time_trace


path, f = os.path.split(os.path.abspath(__file__))


class TestTwoStateExpTruth(test_time_trace.TestTwoStateExpTruth):
    """Test the `TwoStateExpTruth` class"""
    def setUp(self):
        super().setUp()
        self.truth_gen = time_trace_numba.TwoStateExpTruth

    def test_generate(self):
        "time_trace_numba.TwoStateExpTruth.generate: Basic functionality"
        super().test_generate()

    def test_generate_loop(self):
        """time_trace_numba.TwoStateExpTruth.generate: Trigger dynamic list

        Fixed length does not create a simulation that is long enough,
        therefore fall back to dynamic list.
        """
        super().test_generate_loop()

    def test_generate_short(self):
        """time_trace_numba.TwoStateExpTruth.generate: Minimum length

        Make sure at least one data point is generated.
        """
        super().test_generate_short()

    def test_generate_random_seed0(self):
        """time_trace_numba.TwoStateExpTruth.generate: Use RNG with seed 0"""
        @numba.njit
        def seed0():
            # This has to be done within a JITed function
            np.random.seed(0)

        seed0()
        super().test_generate_random_seed0()

    def test_init_list(self):
        """time_trace_numba.TwoStateExpTruth.__init__: Init with lists"""
        tr = self.truth_gen([2., 4.], [0.8, 0.2])
        tr.generate(1)


class TestSample(test_time_trace.TestSample):
    """Test the `sample` function"""
    def setUp(self):
        super().setUp()
        self.sample_func = time_trace_numba.sample

    def test_call(self):
        """time_trace_numba.sample: Basic functionality"""
        super().test_call()

    def test_random_input(self):
        """time_trace_numba.sample: Use output of TwoStateExpTruth as input

        Compare to the saved result of a test run. The test run was not
        verified except using the other tests of this class, so this is more
        a regression test than a functionality test.
        """
        super().test_random_input()

    @unittest.skip("N/A")
    def test_long_trace(self):
        """time_trace_numba.sample: Check for rounding errors in long traces

        Due to implementation differences, this is not a problem for the numba
        based sample function.
        """
        super().test_long_trace()

    def test_frame_time(self):
        """time_trace_numba.sample: Non-zero frame_time"""
        super().test_frame_time()

    def test_random_input_frame_time(self):
        """time_trace_numba.sample: Rand. inp. TwoStateExpTruth (w/ frame_time)

        Compare to the saved result of a test run. The test run was not
        verified except using the other tests of this class, so this is more
        a regression test than a functionality test.
        """
        super().test_random_input_frame_time()


@numba.jitclass([("mult", numba.float64)])
class FluoModel:
    def __init__(self, mult):
        self.mult = mult

    def generate(self, m):
        return self.mult * m


class TestExperiment(test_time_trace.TestExperiment):
    """Test the `experiment` function"""
    def setUp(self):
        super().setUp()
        self.exp_func = time_trace_numba.experiment
        self.donor_gen = FluoModel(3)
        self.acceptor_gen = FluoModel(2)

    def test_call(self):
        """time_trace_numba.experiment: Basic functionality"""
        super().test_call()


class TestSimulateDataset(test_time_trace.TestSimulateDataset):
    """Test the `simulate_dataset` function"""
    def setUp(self):
        super().setUp()
        self.lifetimes = np.array([2., 4.])
        self.eff = np.array([0.8, 0.2])
        self.truth = time_trace_numba.TwoStateExpTruth(self.lifetimes,
                                                       self.eff)
        self.sample_func = time_trace_numba.sample
        self.exp_func = time_trace_numba.experiment
        self.dataset_func = time_trace_numba.simulate_dataset
        self.donor_gen = FluoModel(3)
        self.acceptor_gen = FluoModel(2)

    def test_call(self):
        """time_trace_numba.simulate_dataset: Basic functionality"""
        super().test_call()

    @unittest.skip("N/A")
    def test_truth_array(self):
        """time_trace_numab.simulate_dataset: Pass array as truth parameter

        This is not implemented for the numba-accelerated version. Skip.
        """
        super().test_truth_array()


if __name__ == "__main__":
    unittest.main()
