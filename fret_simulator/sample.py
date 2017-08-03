import multiprocessing

import numpy as np
import scipy

from . import sim_numpy


def sample_p_vals(life_times, efficiencies, sample_rate, data_points, photons,
                  experiment_data, photon_precision=0.1,
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

    if n_cpus <= 1:
        ret = sample_p_vals_numpy(lt, ef, sample_rate, data_points, photons,
                                  experiment_data, lognorm_params,
                                  photon_precision)
    else:
        with multiprocessing.Pool(n_cpus) as pool:
            # split data into `n_cpus` chunks. Crude, but probably sufficient.
            lt_s = np.array_split(lt, n_cpus)
            ef_s = np.array_split(ef, n_cpus)

            ares = []
            for l, e in zip(lt_s, ef_s):
                args = (l, e, sample_rate, data_points, photons,
                        experiment_data, lognorm_params, photon_precision)
                r = pool.apply_async(sample_p_vals_numpy, args)
                ares.append(r)

            ret = np.concatenate([r.get() for r in ares])

    return p_val_from_ks(ret.reshape(shape), data_points, experiment_data.size)


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


def p_val_from_ks(ks, n1, n2):
    f = n1 * n2 / (n1 + n2)
    return scipy.special.kolmogorov(np.sqrt(f) * ks)
