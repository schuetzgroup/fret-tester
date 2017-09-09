import unittest
import os

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

    def test_init_list(self):
        """time_trace_numba.TwoStateExpTruth.__init__: Init with lists"""
        tr = self.truth_gen([2., 4.], [0.8, 0.2])
        tr.generate(1)


class TestSample(test_time_trace.TestSample):
    """Test the `sample` function"""
    def setUp(self):
        self.sample_func = time_trace_numba.sample

    def test_call(self):
        """time_trace_numba.sample: Basic functionality"""
        super().test_call()

    @unittest.skip("N/A")
    def test_long_trace(self):
        """time_trace_numba.sample: Check for rounding errors in long traces

        Due to implementation differences, this is not a problem for the numba
        based sample function.
        """
        super().test_long_trace()


if __name__ == "__main__":
    unittest.main()
