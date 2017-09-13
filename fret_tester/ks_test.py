# Copyright 2017 Lukas Schrangl
"""Tools for comparing simulated smFRET data to experimental data"""
import multiprocessing

import numpy as np
import scipy

from . import time_trace


class BatchTester:
    def __init__(self, truth_generator, sim_func):
        self.truth_generator = truth_generator
        self.sim_func = sim_func

    def batch_test(self, test_times, efficiencies, exposure_time, data_points,
                   photons, experiment_data, donor_brightness,
                   acceptor_brightness,
                   nprocs=1, nchunks=None):
        """Test experiment data against many simulated data sets

        For each set of (test) life times, simulate smFRET events and compare
        to the experiment data using a two-sample KS test. The difference is
        expressed in terms of the p-value.

        Parameters
        ----------
        test_times : array_like
            Life times to test experiment data against. Run simulations for
            each set of test lifes. ``test_times[i]`` has to be an array of
            test times for the i-th state.
        efficiencies : array_like
            Array of FRET efficiencies
        exposure_time : float
            Exposure time as used in the experiment
        data_points : int
            Number of smFRET data points to simulate
        photons : int
            Mean number of photons emitted per exposure
        experiment_data : array_like
            smFRET efficiency values from the experiment
        donor_brightness, acceptor_brightness : callable
            Takes one argument, an array of mean brightness values. For each
            entry `m` it returns a random value drawn from the brightness
            distribution with mean `m`.
        nprocs : int, optional
            Number of processes to use for parallel simulation. Defaults to 1,
            i.e. no multiprocessing, since multiprocessing requires additional
            measures, such as proper ``if __name__ == "__main__"`` blocks.
        nchunks : int or None, optional
            Number of chunks into which to split the data for parallel
            simulation. If `None` (default), use ``nprocs * 2``.

        Returns
        -------
        numpy.ndarray
            p-values returned by KS tests. The array has the same shape as
            the `test_times` minus the first dimension.
        """
        # Transposing is necessary for broadcasting to work in case
        # `efficiencies` is just a 1D array. Additionally, the flattened result
        # can be  passed directly to `batch_test_worker`, which can iterate
        # over it.
        lt, ef = np.broadcast_arrays(np.transpose(test_times),
                                     np.transpose(efficiencies))
        shape = lt.shape[:-1]
        num_states = lt.shape[-1]
        # "Flatten" s.t. each line gives a set of life times (or efficiencies,
        # respectively). This is necessary for the `batch_test_worker` call.
        lt = np.reshape(lt, (-1, num_states), "C")
        ef = np.reshape(ef, (-1, num_states), "C")

        if nprocs <= 1:
            # If no multiprocessing is wanted, just execute the worker function
            ret = self._batch_test_worker(lt, ef, exposure_time, data_points,
                                          photons, experiment_data,
                                          donor_brightness,
                                          acceptor_brightness)
        else:
            # Use multiprocessing
            if not nchunks:
                # This seems to be a smart choice according to conducted
                # benchmarks
                nchunks = nprocs * 2
            # Use the "spawn" method to create processes. At any rate, don't
            # use  "fork", since children inherit the parent's RNG state and
            # every  child process will generate the same random numbers.
            with multiprocessing.get_context("spawn").Pool(nprocs) as pool:
                # Split data into `nchunks` chunks.
                lt_s = np.array_split(lt, nchunks)
                ef_s = np.array_split(ef, nchunks)

                ares = []
                for l, e in zip(lt_s, ef_s):
                    # Async call to the worker function in the pool's processes
                    args = (l, e, exposure_time, data_points, photons,
                            experiment_data, donor_brightness,
                            acceptor_brightness)
                    r = pool.apply_async(self._batch_test_worker, args)
                    # Keep the async call results
                    ares.append(r)

                # Wait for the results and concatenate them
                ret = np.concatenate([r.get() for r in ares])

        # Reshape to original shape and transpose again to undo intial
        # transpose
        return np.reshape(ret, shape).T

    def frame_time(self, life_times):
        return 2 * max(life_times)

    def _batch_test_worker(self, test_times, efficiencies, exposure_time,
                           data_points, photons, experiment_data,
                           donor_brightness, acceptor_brightness):
        """Test experiment data against many simulated data sets (worker func)

        For each set of (test) life times, simulate smFRET events and compare
        to the experiment data using a two-sample KS test. The difference is
        expressed in terms of the KS statistics.

        Parameters
        ----------
        test_times : array_like, shape(n, 2)
            Life times to test experiment data against. Run simulations for
            each set of test lifes. ``test_times[i]`` a set of test life times
            (one per state). This is different (transposed) from the
            `test_times` argument of :py:meth:`batch_test`.
        efficiencies : array_like, shape(n, 2)
            Array of FRET efficiencies. Has to be of the same shape as
            `test_times`.
        exposure_time : float
            Exposure time as used in the experiment
        data_points : int
            Number of smFRET data points to simulate
        photons : int
            Mean number of photons emitted per exposure
        experiment_data : array_like
            smFRET efficiency values from the experiment
        donor_brightness, acceptor_brightness : callable
            Takes one argument, an array of mean brightness values. For each
            entry `m` it returns a random value drawn from the brightness
            distribution with mean `m`.

        Returns
        -------
        list
            p-values returned by KS tests.
        """
        ret = []
        for lt, ef in zip(test_times, efficiencies):
            truth = self.truth_generator(lt, ef)
            d = self.sim_func(truth, exposure_time, data_points, photons,
                              donor_brightness, acceptor_brightness,
                              frame_time=self.frame_time(lt))
            ks, p = scipy.stats.ks_2samp(d.exp_eff, experiment_data)
            ret.append(p)
        return ret


numpy_tester = BatchTester(time_trace.TwoStateExpTruth,
                           time_trace.simulate_dataset)


def batch_test(*args, **kwargs):
    return numpy_tester.batch_test(*args, **kwargs)


def combine_test_results(results):
    n = len(results)
    return 1 - (1 - np.min(results, axis=0))**n
