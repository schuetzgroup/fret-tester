"""Analyze smFRET kinetics using comprehensive MC simulations and testing"""
# Copyright 2017-2018 Lukas Schrangl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from .plot import Plotter1D, Plotter2D
from .fluo_models import LognormalBrightness, PolyLnBrightness
from .time_trace import (TwoStateExpTruth, sample, experiment,
                         DataSet, simulate_dataset)
from .ks_test import (BatchTester, numpy_tester, batch_test,
                      combine_test_results)
from .utils import bi_beta_fit

from ._version import __version__
