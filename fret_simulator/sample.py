import multiprocessing

import numpy as np
import numba
import scipy

from . import sim_numba, sim_numpy, ks_numba


def sample_p_vals(life_times, efficiencies, sample_rate, data_points, photons,
                  experiment_data, photon_precision=0.1, engine="numba",
                  n_cpus=multiprocessing.cpu_count()):
    lt, ef = np.broadcast_arrays(life_times, efficiencies)
    shape = lt.shape[:-1]
    num_states = lt.shape[-1]
    lt = np.reshape(lt, (-1, num_states), "C")
    ef = np.reshape(ef, (-1, num_states), "C")

    # pre-calculate logarithms
    if photon_precision > 0:
        lognorm_params = sim_numpy.lognormal_parms(
                np.arange(0, photons+photon_precision/2, photon_precision))
        lognorm_params = np.array(lognorm_params).T
    else:
        lognorm_params = np.empty((0, 2))

    if engine == "numpy":
        work_func = sample_p_vals_numpy
    elif engine == "numba":
        work_func = sample_p_vals_numba
    else:
        raise ValueError('engine has to be one of {"numpy", "numba"}')

    if n_cpus <= 1:
        ret = work_func(lt, ef, sample_rate, data_points, photons,
                        experiment_data, lognorm_params, photon_precision)
    else:
        with multiprocessing.Pool(n_cpus) as pool:
            # split data into `n_cpus` chunks. Crude, but probably sufficient.
            lt_s = np.array_split(lt, n_cpus)
            ef_s = np.array_split(ef, n_cpus)

            ares = []
            for l, e in zip(lt_s, ef_s):
                args = (l, e, sample_rate, data_points, photons,
                        experiment_data, lognorm_params, photon_precision)
                r = pool.apply_async(work_func, args)
                ares.append(r)

            ret = np.concatenate([r.get() for r in ares])

    return ks_to_p_val(ret.reshape(shape), data_points, experiment_data.size)


def sample_p_vals_numpy(life_times, efficiencies, sample_rate, data_points,
                        photons, experiment_data, lognorm_params,
                        photon_precision):
    ret = np.empty(life_times.shape[0], dtype=float)
    dur = data_points / sample_rate
    for i, (lt, ef) in enumerate(zip(life_times, efficiencies)):
        truth = sim_numpy.two_state_truth(lt, ef, dur)
        st, se = sim_numpy.sample(*truth, data_points, sample_rate)
        d, a = sim_numpy.experiment(se, photons, lognorm_params,
                                    photon_precision)
        e = a / (d+a)
        ks, p = scipy.stats.ks_2samp(e, experiment_data)
        ret[i] = ks
    return ret


@numba.guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:],
                     numba.int64[:], numba.float64[:], numba.float64[:],
                     numba.float64[:, :], numba.float64[:], numba.float64[:])],
                   "(n),(n),(),(),(),(l),(m, j),()->()",
                   nopython=True, target="cpu")
def sample_p_vals_numba(life_times, efficiencies, sample_rate, data_points,
                        photons, experiment_data, lognorm_params,
                        photon_precision, ret):
    tt, te = sim_numba.two_state_truth(life_times, efficiencies,
                                       data_points[0]/sample_rate[0])
    st, se = sim_numba.sample(tt, te, data_points[0], sample_rate[0])
    d, a = sim_numba.experiment(se, photons[0], lognorm_params,
                                photon_precision[0])
    e = np.empty_like(d)
    for i in range(data_points[0]):
        e[i] = a[i] / (a[i] + d[i])
    ret[0] = ks_numba.ks_2samp(e, experiment_data)


def ks_to_p_val(ks, n1, n2):
    f = n1 * n2 / (n1 + n2)
    return scipy.special.kolmogorov(np.sqrt(f) * ks)
