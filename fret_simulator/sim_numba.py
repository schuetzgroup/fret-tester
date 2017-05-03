import numpy as np
import numba

from .numpy_sim import photon_std_from_mean


@numba.jit(nopython=True)
def two_state_truth(lifetimes, efficiencies, duration, max_retries=10):
    num_events = int(duration / np.mean(lifetimes) * 1.5)
    #  num_events increased by 50% to be sure

    for rt in range(max_retries):
        t = 0.
        i = 0
        s = np.random.rand() < lifetimes[0]/np.sum(lifetimes)
        ret_t = np.full(num_events, np.NaN)
        ret_e = np.full(num_events, np.NaN)
        while i < num_events:
            t += np.random.exponential(lifetimes[int(s)])
            ret_t[i] = t
            ret_e[i] = efficiencies[int(s)]
            s = not s
            i += 1
            if t > duration:
                return ret_t, ret_e

    raise RuntimeError("Failed to generate truth.")


@numba.jit(nopython=True)
def sample(time, eff, data_points, sample_rate):
    ret_t = np.empty(data_points)
    ret_e = np.empty(data_points)
    sample_t = 1. / sample_rate

    t_idx = 0
    for dp in range(0, data_points):
        t_end = (dp + 1) * sample_t
        ret_t[dp] = t_end

        intens = 0.
        t_sub = t_end - sample_t
        while time[t_idx] < t_end:
            intens += (time[t_idx] - t_sub) * eff[t_idx]
            t_sub = time[t_idx]
            t_idx += 1
        intens += (t_end - t_sub) * eff[t_idx]

        ret_e[dp] = intens * sample_rate
    return ret_t, ret_e


photon_std_from_mean_nb = numba.jit(photon_std_from_mean, nopython=True)


@numba.jit(nopython=True)
def lognormal_parms(m):
    s = photon_std_from_mean_nb(m)
    mu = np.log(m / np.sqrt(1 + s**2/m**2))
    sigma = np.sqrt(np.log(1 + s**2/m**2))
    return mu, sigma


@numba.jit(nopython=True)
def lognormal(m):
    mu, sigma = lognormal_parms(m)
    return np.random.lognormal(mu, sigma)


@numba.jit(nopython=True)
def experiment(eff, photons):
    data_points = len(eff)
    ret_a = np.empty(data_points)
    ret_d = np.empty(data_points)
    for i in range(data_points):
        e = eff[i]
        ret_d[i] = lognormal((1 - e) * photons)
        ret_a[i] = lognormal(e * photons)
    return ret_d, ret_a
