import unittest
import os

import numba

from fret_tester import time_trace_numba
from . import test_time_trace


path, f = os.path.split(os.path.abspath(__file__))


class TestTwoStateExpTruth(test_time_trace.TestTwoStateExpTruth):
    """Test the `TwoStateExpTruth` class"""
    def setUp(self):
        super().setUp()
        self.truth_gen = time_trace_numba.TwoStateExpTruth

    def test_call(self):
        "time_trace_numba.TwoStateExpTruth.__call__: Basic functionality"
        super().test_call()

    def test_call_loop(self):
        """time_trace_numba.TwoStateExpTruth.__call__: Trigger dynamic list

        Fixed length does not create a simulation that is long enough,
        therefore fall back to dynamic list.
        """
        super().test_call_loop()

    def test_call_short(self):
        """time_trace_numba.TwoStateExpTruth.__call__: Minimum length

        Make sure at least one data point is generated.
        """
        super().test_call_short()

    def test_init_list(self):
        """time_trace_numba.TwoStateExpTruth.__init__: Init with lists"""
        tr = self.truth_gen([2., 4.], [0.8, 0.2])
        tr.generate(1)


if __name__ == "__main__":
    unittest.main()
