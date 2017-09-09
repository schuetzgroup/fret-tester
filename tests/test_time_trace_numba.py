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


if __name__ == "__main__":
    unittest.main()
