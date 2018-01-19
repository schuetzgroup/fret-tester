import unittest
import os

import numpy as np

from fret_tester import fluo_models


path, f = os.path.split(os.path.abspath(__file__))


def make_lognormal_seed0():
    """Generate "lognormal_seed0.npz"

    Used for tests.
    """
    m = np.linspace(1, 10, 91)
    mu, sigma = LnBr()._parameters(m)

    np.random.seed(0)
    b = np.random.lognormal(mu, sigma)

    np.savez_compressed(os.path.join(path, "lognormal_seed0.npz"), b=b)


class LnBr(fluo_models.LognormalBrightness):
    def std_from_mean(self, m):
        return m / 2 + 1


class TestLognormalBrightness(unittest.TestCase):
    """Test the `TwoStateExpTruth` class"""
    def setUp(self):
        self.LnBrClass = LnBr

    def test_parameters(self):
        """fluo_models.LognormalBrightness.parameters"""
        lnb = self.LnBrClass(0., 0.)

        m = np.arange(1, 21)
        exp_mu, exp_sig = lnb.parameters(m)

        s = lnb.std_from_mean(m)
        x = 1 + s**2 / m**2

        np.testing.assert_allclose(exp_mu, np.log(m / np.sqrt(x)))
        np.testing.assert_allclose(exp_sig, np.sqrt(np.log(x)))

    def test_init_parameter_cache(self):
        """fluo_models.LognormalBrightness.__init__: Caching of parameters"""
        prec = 0.12
        max_br = 982

        lnb = self.LnBrClass(max_br, prec)

        m = np.array([i * prec for i in range(int(max_br / prec) + 2)])
        p = np.asarray(lnb.parameters(m)).T

        np.testing.assert_equal(lnb._cached_precision, prec)
        np.testing.assert_allclose(lnb._cached_params, p)

    def test_generate(self):
        """fluo_models.LognormalBrightness.generate"""
        lnb = self.LnBrClass(0., 0.)
        lnb._test = 1
        m = np.linspace(1., 10., 91)
        mu, sigma = lnb.parameters(m)

        r = lnb.generate(m)
        np.testing.assert_allclose(r, mu * sigma)

    def test_generate_cache(self):
        """fluo_models.LognormalBrightness.generate: Use cached parameters"""
        lnb = self.LnBrClass(10., 1.)
        lnb._test = 1
        m = np.linspace(1., 10., 91)
        m_rounded = np.round(m)
        mu, sigma = lnb.parameters(m_rounded)

        r = lnb.generate(m)
        np.testing.assert_allclose(r, mu * sigma)

    def test_generate_random_seed0(self):
        """fluo_models.LognormalBrightness.generate: Use RNG with seed 0"""
        lnb = self.LnBrClass(0., 0.)
        np.random.seed(0)
        m = np.linspace(1., 10., 91)
        b = lnb.generate(m)

        des = np.load(os.path.join(path, "lognormal_seed0.npz"))["b"]
        np.testing.assert_allclose(b, des)


class TestPolyLnBrightness(TestLognormalBrightness):
    def setUp(self):
        super().setUp()
        self.LnBrClass = lambda t, e: fluo_models.PolyLnBrightness([0.5, 1],
                                                                   t, e)

    def test_parameters(self):
        """fluo_models.PolyLnBrightness.parameters"""
        super().test_parameters()

    def test_init_parameter_cache(self):
        """fluo_models.PolyLnBrightness.__init__: Caching of parameters"""
        super().test_init_parameter_cache()

    def test_generate(self):
        """fluo_models.PolyLnBrightness.generate"""
        super().test_generate()

    def test_generate_cache(self):
        """fluo_models.PolyLnBrightness.generate: Use cached parameters"""
        super().test_generate_cache()

    def test_generate_random_seed0(self):
        """fluo_models.PolyLnBrightness.generate: Use RNG with seed 0"""
        super().test_generate_random_seed0()


if __name__ == "__main__":
    unittest.main()
