import multiprocessing

import numpy as np
import numba
import scipy

from . import sim_numba, sim_numpy, ks_numba


def sample_p_vals(life_times, efficiencies, sample_rate, data_points, photons,
                  experiment_data, engine="numpy"):
    lt, ef = np.broadcast_arrays(life_times.T, efficiencies)
    shape = lt.shape[:-1]
    num_states = lt.shape[-1]
    lt = np.reshape(lt, (-1, num_states), "C")
    ef = np.reshape(ef, (-1, num_states), "C")

    if engine == "numpy":
        ret = sample_p_vals_numpy(lt, ef, sample_rate, data_points, photons,
                                  experiment_data)
    elif engine == "numba":
        ret = sample_p_vals_numba(lt, ef, sample_rate, data_points, photons,
                                  experiment_data)
    elif engine == "mp":
        ret = sample_p_vals_mp(lt, ef, sample_rate, data_points, photons,
                               experiment_data)
    else:
        raise ValueError('engine has to be one of {"numpy", "numba"}')

    return ks_to_p_val(ret.reshape(shape), data_points, experiment_data.size)


def sample_p_vals_numpy(life_times, efficiencies, sample_rate, data_points,
                        photons, experiment_data):
    ret = np.empty(life_times.shape[0], dtype=float)
    dur = data_points / sample_rate
    for i, (lt, ef) in enumerate(zip(life_times, efficiencies)):
        truth = sim_numpy.two_state_truth(lt, ef, dur)
        st, se = sim_numpy.sample(*truth, data_points, sample_rate)
        d, a = sim_numpy.experiment(se, photons)
        e = a / (d+a)
        ks, p = scipy.stats.ks_2samp(e, experiment_data)
        ret[i] = ks
    return ret


@numba.guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:],
                     numba.int64[:], numba.float64[:], numba.float64[:],
                     numba.float64[:])],
                   "(n),(n),(),(),(),(l)->()", nopython=True, target="cpu")
def sample_p_vals_numba(life_times, efficiencies, sample_rate, data_points,
                        photons, experiment_data, ret):
    tt, te = sim_numba.two_state_truth(life_times, efficiencies,
                                       data_points[0]/sample_rate[0])
    st, se = sim_numba.sample(tt, te, data_points[0], sample_rate[0])
    d, a = sim_numba.experiment(se, photons[0])
    e = np.empty_like(d)
    for i in range(data_points[0]):
        e[i] = a[i] / (a[i] + d[i])
    ret[0] = ks_numba.ks_2samp(e, experiment_data)


def sample_p_vals_mp(life_times, efficiencies, sample_rate, data_points,
                     photons, experiment_data):
    n_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpus)

    # split data into `n_cpus` chunks. Maybe this can be optimized.
    lt = np.array_split(life_times, n_cpus)
    ef = np.array_split(efficiencies, n_cpus)

    ares = []
    for l, e in zip(lt, ef):
        args = (l, e, sample_rate, data_points, photons, experiment_data)
        r = pool.apply_async(sample_p_vals_numba, args)
        ares.append(r)

    return np.concatenate([r.get() for r in ares])


def ks_to_p_val(ks, n1, n2):
    f = n1 * n2 / (n1 + n2)
    return scipy.special.kolmogorov(np.sqrt(f) * ks)
