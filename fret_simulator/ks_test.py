# Copyright 2017 Lukas Schrangl
"""Tools for comparing simulated smFRET data to experimental data"""
import multiprocessing

import numpy as np
import scipy

from . import time_trace


def batch_test(test_times, efficiencies, exposure_time, data_points,
               photons, experiment_data, donor_brightness,
               acceptor_brightness,
               nproc=multiprocessing.cpu_count()):
    """Test experiment data against many simulated data sets

    For each set of (test) life times, simulate smFRET events and compare to
    the experiment data using a two-sample KS test. The difference is
    expressed in terms of the p-value.

    Parameters
    ----------
    test_times : array_like
        Life times to test experiment data against. Run simulations for each
        set of test lifes. ``test_times[..., i]`` has to be an array of test
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
    nproc : int, optional
        Number of processes to use for parallel simulation. Defaults to the
        number of CPUs.

    Returns
    -------
    numpy.ndarray
        p-values returned by KS tests. The array has the same shape as
        the `test_times` minus the last dimension.
    """
    lt, ef = np.broadcast_arrays(test_times, efficiencies)
    shape = lt.shape[:-1]
    num_states = lt.shape[-1]
    lt = np.reshape(lt, (-1, num_states), "C")
    ef = np.reshape(ef, (-1, num_states), "C")

    if nproc <= 1:
        ret = batch_test_worker(lt, ef, exposure_time, data_points, photons,
                                experiment_data, donor_brightness,
                                acceptor_brightness)
    else:
        with multiprocessing.Pool(nproc) as pool:
            # split data into `n_cpus` chunks. Crude, but probably sufficient.
            lt_s = np.array_split(lt, nproc)
            ef_s = np.array_split(ef, nproc)

            ares = []
            for l, e in zip(lt_s, ef_s):
                args = (l, e, exposure_time, data_points, photons,
                        experiment_data, donor_brightness, acceptor_brightness)
                r = pool.apply_async(batch_test_worker, args)
                ares.append(r)

            ret = np.concatenate([r.get() for r in ares])

    return p_val_from_ks(ret.reshape(shape), data_points, experiment_data.size)


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
        per state).
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
    numpy.ndarray
        KS statics returned by KS tests
    """
    ret = np.empty(test_times.shape[0], dtype=float)
    dur = data_points * exposure_time
    for i, (lt, ef) in enumerate(zip(test_times, efficiencies)):
        truth = time_trace.two_state_truth(lt, ef, dur)
        st, se = time_trace.sample(*truth, exposure_time, data_points)
        d, a = time_trace.experiment(se, photons, donor_brightness,
                                     acceptor_brightness)
        e = a / (d+a)
        ks, p = scipy.stats.ks_2samp(e, experiment_data)
        ret[i] = ks
    return ret


def p_val_from_ks(ks, n1, n2):
    """Calculate p-value from KS statistics

    resulting from two-sample KS tests.

    Parameters
    ----------
    ks : array_like
        KS statistics values
    n1, n2 : int
        Number of data points in the data sets

    Returns
    -------
    numpy.ndarray
        p-values corresponding to KS statistic values
    """
    f = n1 * n2 / (n1 + n2)
    return scipy.special.kolmogorov(np.sqrt(f) * ks)
