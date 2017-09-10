import unittest
import os

import numba
import numpy as np

from fret_tester import fluo_models_numba
from . import test_fluo_models


path, f = os.path.split(os.path.abspath(__file__))


@fluo_models_numba.lognormal_jitclass
class LnBr(fluo_models_numba.LognormalBrightness):
    def std_from_mean(self, m):
        return m / 2 + 1


class TestLognormalBrightness(test_fluo_models.TestLognormalBrightness):
    """Test the `TwoStateExpTruth` class"""
    def setUp(self):
        self.LnBrClass = LnBr

    def test_parameters(self):
        """fluo_models_numba.LognormalBrightness._parameters"""
        super().test_parameters()

    def test_init_parameter_cache(self):
        """fluo_models_numba.LognormalBrightness.__init__: Caching of params"""
        super().test_init_parameter_cache()

    def test_generate(self):
        """fluo_models_numba.LognormalBrightness.generate"""
        super().test_generate()

    def test_generate_cache(self):
        """fluo_models_numba.LognormalBrightness.generate: Use cached params"""
        super().test_generate_cache()

    def test_generate_random_seed0(self):
        """fluo_models_numba.LognormalBrightness.generate: Use RNG w/ seed 0"""
        @numba.njit
        def seed0():
            # This has to be done within a JITed function
            np.random.seed(0)

        seed0()
        super().test_generate_random_seed0()


if __name__ == "__main__":
    unittest.main()
