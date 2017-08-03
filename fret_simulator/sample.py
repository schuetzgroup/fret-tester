import multiprocessing

import numpy as np
import scipy

from . import time_trace


def sample_p_vals(life_times, efficiencies, exposure_time, data_points,
                  photons, experiment_data, donor_brightness,
                  acceptor_brightness,
                  n_cpus=multiprocessing.cpu_count()):
    lt, ef = np.broadcast_arrays(life_times, efficiencies)
    shape = lt.shape[:-1]
    num_states = lt.shape[-1]
    lt = np.reshape(lt, (-1, num_states), "C")
    ef = np.reshape(ef, (-1, num_states), "C")

    if n_cpus <= 1:
        ret = sample_p_vals_numpy(lt, ef, exposure_time, data_points, photons,
                                  experiment_data, donor_brightness,
                                  acceptor_brightness)
    else:
        with multiprocessing.Pool(n_cpus) as pool:
            # split data into `n_cpus` chunks. Crude, but probably sufficient.
            lt_s = np.array_split(lt, n_cpus)
            ef_s = np.array_split(ef, n_cpus)

            ares = []
            for l, e in zip(lt_s, ef_s):
                args = (l, e, exposure_time, data_points, photons,
                        experiment_data, donor_brightness, acceptor_brightness)
                r = pool.apply_async(sample_p_vals_numpy, args)
                ares.append(r)

            ret = np.concatenate([r.get() for r in ares])

    return p_val_from_ks(ret.reshape(shape), data_points, experiment_data.size)


def sample_p_vals_numpy(life_times, efficiencies, exposure_time, data_points,
                        photons, experiment_data, donor_brightness,
                        acceptor_brightness):
    ret = np.empty(life_times.shape[0], dtype=float)
    dur = data_points * exposure_time
    for i, (lt, ef) in enumerate(zip(life_times, efficiencies)):
        truth = time_trace.two_state_truth(lt, ef, dur)
        st, se = time_trace.sample(*truth, exposure_time, data_points)
        d, a = time_trace.experiment(se, photons, donor_brightness,
                                     acceptor_brightness)
        e = a / (d+a)
        ks, p = scipy.stats.ks_2samp(e, experiment_data)
        ret[i] = ks
    return ret


def p_val_from_ks(ks, n1, n2):
    f = n1 * n2 / (n1 + n2)
    return scipy.special.kolmogorov(np.sqrt(f) * ks)
