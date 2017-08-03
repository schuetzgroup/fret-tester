# Copyright 2017 Lukas Schrangl
"""Tools for comparing simulated smFRET data to experimental data"""
import multiprocessing

import numpy as np
import scipy

from . import time_trace


def batch_test(test_times, efficiencies, exposure_time, data_points,
               photons, experiment_data, donor_brightness,
               acceptor_brightness,
               nprocs=multiprocessing.cpu_count(), nchunks=None):
    """Test experiment data against many simulated data sets

    For each set of (test) life times, simulate smFRET events and compare to
    the experiment data using a two-sample KS test. The difference is
    expressed in terms of the p-value.

    Parameters
    ----------
    test_times : array_like
        Life times to test experiment data against. Run simulations for each
        set of test lifes. ``test_times[i]`` has to be an array of test
        times for the i-th state.
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
        Takes one argument, an array of mean brightness values. For each entry
        `m` it returns a random value drawn from the brightness distribution
        with mean `m`.
    nprocs : int, optional
        Number of processes to use for parallel simulation. Defaults to the
        number of CPU threads.
    nprocs : int or None, optional
        Number of chunks into which to split the data for parallel
        simulation. If `None` (default), use ``nprocs * 2``.

    Returns
    -------
    numpy.ndarray
        p-values returned by KS tests. The array has the same shape as
        the `test_times` minus the first dimension.
    """
    lt, ef = np.broadcast_arrays(np.transpose(test_times),
                                 np.transpose(efficiencies))
    shape = lt.shape[:-1]
    num_states = lt.shape[-1]
    lt = np.reshape(lt, (-1, num_states), "C")
    ef = np.reshape(ef, (-1, num_states), "C")

    if nprocs <= 1:
        ret = batch_test_worker(lt, ef, exposure_time, data_points, photons,
                                experiment_data, donor_brightness,
                                acceptor_brightness)
    else:
        if not nchunks:
            # This seems to be a smart choice according to conducted
            # benchmarks
            nchunks = nprocs * 2
        with multiprocessing.Pool(nprocs) as pool:
            # split data into `nchunks` chunks.
            lt_s = np.array_split(lt, nchunks)
            ef_s = np.array_split(ef, nchunks)

            ares = []
            for l, e in zip(lt_s, ef_s):
                args = (l, e, exposure_time, data_points, photons,
                        experiment_data, donor_brightness, acceptor_brightness)
                r = pool.apply_async(batch_test_worker, args)
                ares.append(r)

            ret = np.concatenate([r.get() for r in ares])

    return np.reshape(ret, shape).T


def batch_test_worker(test_times, efficiencies, exposure_time, data_points,
                      photons, experiment_data, donor_brightness,
                      acceptor_brightness):
    """Test experiment data against many simulated data sets (worker function)

    For each set of (test) life times, simulate smFRET events and compare to
    the experiment data using a two-sample KS test. The difference is
    expressed in terms of the KS statistics.

    Parameters
    ----------
    test_times : array_like, shape=(n, 2)
        Life times to test experiment data against. Run simulations for each
        set of test lifes. ``test_times[i]`` a set of test life times (one
        per state). This is different (transposed) from the `test_times`
        argument of :py:func:`batch_test`.
    efficiencies : array_like, shape=(n, 2)
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
        Takes one argument, an array of mean brightness values. For each entry
        `m` it returns a random value drawn from the brightness distribution
        with mean `m`.

    Returns
    -------
    list
        p-values returned by KS tests.
    """
    ret = []
    for lt, ef in zip(test_times, efficiencies):
        d = time_trace.simulate_dataset(lt, ef, exposure_time, data_points,
                                        photons, donor_brightness,
                                        acceptor_brightness)
        ks, p = scipy.stats.ks_2samp(d.exp_eff, experiment_data)
        ret.append(p)
    return ret
