import bandits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def q1(time_horizon=1000):
    rewards_ucb, draws_ucb = bandit.UCB1(time_horizon)
    plt.figure()
    draws_ucb.plot(style='.', title='UCB1')

    rewards_ts, draws_ts = bandit.TS(time_horizon)
    plt.figure()
    draws_ts.plot(style='.', title='Thompson Sampling')


def q2(bandit=bandits.Bernoulli(), n_samples=100, time_horizon=1000, figtitle="regret_curves"):
    p_star = max(bandit.means)
    regret_ucb = pd.Series(np.zeros(time_horizon))
    regret_ts = pd.Series(np.zeros(time_horizon))
    regret_naive = pd.Series(np.zeros(time_horizon))

    for _ in range(n_samples):
        rewards_ucb, draws_ucb = bandit.UCB1(time_horizon)
        rewards_ts, draws_ts = bandit.TS(time_horizon)
        rewards_naive, draws_naive = bandit.naive(time_horizon)

        regret_ucb -= rewards_ucb.cumsum()
        regret_ts -= rewards_ts.cumsum()
        regret_naive -= rewards_naive.cumsum()

    regret_ucb /= n_samples
    regret_ts /= n_samples
    regret_naive /= n_samples

    regret_ucb += pd.Series(np.linspace(1, time_horizon, time_horizon)) * p_star
    regret_ts += pd.Series(np.linspace(1, time_horizon, time_horizon)) * p_star
    regret_naive += pd.Series(np.linspace(1, time_horizon, time_horizon)) * p_star

    regret_oracle = pd.Series([bandit.LRlowerbound() * np.log(t) for t in range(time_horizon)])

    fig = plt.figure()
    regret_ucb.plot(label='UCB regret')
    regret_ts.plot(label='Thompson Sampling regret')
    regret_naive.plot(label='Naive algorithm regret')
    regret_oracle.plot(label='Oracle regret')

    plt.legend()
    plt.title('Regret curves')
    fig.savefig(figtitle + ".png")
