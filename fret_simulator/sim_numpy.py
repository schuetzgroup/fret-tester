import numpy as np
import scipy


def two_state_truth(lifetimes, efficiencies, duration):
    num_states = len(lifetimes)
    num_events = int(duration / np.mean(lifetimes) * 1.5) // num_states
    #  num_events increased by 50% to be sure

    prob_1 = lifetimes[0] / np.sum(lifetimes)
    time = []
    eff = []
    dur = 0
    while dur < duration:
        start_with_1 = np.random.rand() < prob_1

        t = np.random.exponential(lifetimes, (num_events, num_states))
        e = np.broadcast_to(efficiencies, (num_events, num_states))
        if not start_with_1:
            t = t[:, ::-1]
            e = e[:, ::-1]

        t = np.cumsum(t.flatten())
        if time:
            t += time[-1][-1]
        time.append(t)
        dur = t[-1]

        eff.append(e.flatten())

    time = np.concatenate(time)
    eff = np.concatenate(eff)

    long_idx = np.nonzero(time > duration)[0][0] + 1
    return time[:long_idx], eff[:long_idx]


def make_step_function(x, y):
    x2 = np.roll(np.repeat(x, 2), 1)
    x2[0] = 0
    return x2, np.repeat(y, 2)


def sample(time, eff, data_points, sample_rate):
    step_t, step_eff = make_step_function(time, eff)
    int_eff = scipy.integrate.cumtrapz(step_eff, step_t, initial=0)
    t_step = 1/sample_rate
    # sample_t = np.arange(0, duration + 0.1*t_step, t_step)
    sample_t = np.linspace(0, data_points*t_step, data_points+1, endpoint=True)
    sample_int_eff = np.interp(sample_t, step_t, int_eff,
                               left=np.NaN, right=np.NaN)
    return sample_t[1:], np.diff(sample_int_eff) * sample_rate


def photon_std_from_mean(m):
    # return 0.22 * m + 35
    return 0.47273326 * m + 6.10658777


def lognormal_parms(m):
    s = photon_std_from_mean(m)

    mu = np.empty_like(m, dtype=float)
    sigma = np.empty_like(s, dtype=float)

    gt0 = m > 0
    m_gt0 = m[gt0]
    s_gt0 = s[gt0]

    x = 1 + s_gt0**2 / m_gt0**2
    mu[gt0] = np.log(m_gt0 / np.sqrt(x))
    sigma[gt0] = np.sqrt(np.log(x))
    mu[~gt0] = -np.inf
    sigma[~gt0] = 0

    return mu, sigma


def lognormal(m, params=np.empty((0, 2)), precision=0.):
    if not (precision and params.size):
        mu, sigma = lognormal_parms(m)
    else:
        idx = np.round(m / precision).astype(int)
        mu, sigma = params[idx].T
    return np.random.lognormal(mu, sigma)


def experiment(eff, photons, ln_params=np.empty((0, 2)), precision=0.):
    acc_p = eff * photons
    don_p = (1 - eff) * photons
    acc_p_noisy = lognormal(acc_p, ln_params, precision)
    don_p_noisy = lognormal(don_p, ln_params, precision)
    return don_p_noisy, acc_p_noisy
