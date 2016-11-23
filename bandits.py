import arms
import pandas as pd
import numpy as np
from scipy.stats import beta, uniform


class Bandit:
    """Class implementing a generic Bandit. Can be constructed from a list of arms or
    as a sum of Bandits. All the following classes inherit from the generic Bandit."""

    def __init__(self, arms=None):
        if arms is not None:
            self.arms = arms
            self.n_arms = len(arms)

    def __add__(self, other):
        return Bandit(arms=self.arms + other.arms)

    def sample(self, k):
        return self.arms[k].sample()

    def UCB1(self, time_horizon=100):
        rews = pd.Series(np.zeros(time_horizon))
        draws = pd.Series(np.zeros(time_horizon))
        draws_per_bandit = pd.Series(np.zeros(self.n_arms))
        means = pd.Series(np.zeros(self.n_arms))
        for t in range(time_horizon):
            if t < self.n_arms:
                arm_to_play = t
            else:
                arm_to_play = (means + np.sqrt(np.log(t) / (2 * draws_per_bandit))).argmax()

            rews.loc[t] = self.sample(arm_to_play)
            draws_per_bandit.loc[arm_to_play] += 1
            means.loc[arm_to_play] = ((draws_per_bandit.loc[arm_to_play] - 1) * means.loc[arm_to_play] + rews.loc[
                t]) / draws_per_bandit.loc[arm_to_play]
            draws.loc[t] = arm_to_play

        return rews, pd.Series(draws)

    def naive(self, time_horizon=100):
        rews = pd.Series(np.zeros(time_horizon))
        draws = pd.Series(np.zeros(time_horizon))
        draws_per_bandit = pd.Series(np.zeros(self.n_arms))
        means = pd.Series(np.zeros(self.n_arms))
        for t in range(time_horizon):
            if t < self.n_arms:
                arm_to_play = t
            else:
                arm_to_play = means.argmax()
            rews.loc[t] = self.sample(arm_to_play)
            draws_per_bandit.loc[arm_to_play] += 1
            means.loc[arm_to_play] = ((draws_per_bandit.loc[arm_to_play] - 1) * means.loc[arm_to_play] + rews.loc[
                t]) / draws_per_bandit.loc[arm_to_play]
            draws.loc[t] = arm_to_play

        return rews, draws

    @staticmethod
    def kl(x, y):
        return x * np.log(y) + (1 - x) * np.log((1 - x) / (1 - y))


class BanditBernoulli(Bandit):
    def __init__(self, means=None, n_arms=None):
        Bandit.__init__(self)
        if means is None:
            if n_arms is None:
                self.n_arms = 10
            else:
                self.n_arms = n_arms
            self.means = [1 / (k + 1) for k in range(1, self.n_arms + 1)]
        else:
            self.means = means
            self.n_arms = len(self.means)
        self.arms = [arms.ArmBernoulli(p) for p in self.means]

    def __repr__(self):
        return self.means.__repr__()

    def TS(self, time_horizon=100):
        rews = pd.Series(np.zeros(time_horizon))
        rew_sum_per_bandit = pd.DataFrame(np.zeros(self.n_arms))
        draws = pd.Series(np.zeros(time_horizon))
        draws_per_bandit = pd.Series(np.zeros(self.n_arms))
        posteriors = [uniform(0, 1) for _ in range(self.n_arms)]
        for t in range(time_horizon):
            if t < self.n_arms:
                arm_to_play = t
            else:
                arm_to_play = pd.Series([pos.rvs() for pos in posteriors]).argmax()

            rews.loc[t] = self.sample(arm_to_play)
            rew_sum_per_bandit.loc[arm_to_play] += rews.loc[t]
            draws_per_bandit.loc[arm_to_play] += 1
            draws.loc[t] = arm_to_play

            posteriors[arm_to_play] = beta(rew_sum_per_bandit.loc[arm_to_play] + 1,
                                           draws_per_bandit.loc[arm_to_play] - rew_sum_per_bandit.loc[arm_to_play] + 1)

        return rews, pd.Series(draws)

    def naive(self, time_horizon=100):
        rews = pd.Series(np.zeros(time_horizon))
        draws = pd.Series(np.zeros(time_horizon))
        draws_per_bandit = pd.Series(np.zeros(self.n_arms))
        means = pd.Series(np.zeros(self.n_arms))
        for t in range(time_horizon):
            if t < self.n_arms:
                arm_to_play = t
            else:
                arm_to_play = means.argmax()
            rews.loc[t] = self.sample(arm_to_play)
            draws_per_bandit.loc[arm_to_play] += 1
            means.loc[arm_to_play] = ((draws_per_bandit.loc[arm_to_play] - 1) * means.loc[arm_to_play] + rews.loc[
                t]) / draws_per_bandit.loc[arm_to_play]
            draws.loc[t] = arm_to_play

        return rews, draws

    def complexity(self):
        p_star = max(self.means)
        return pd.Series([(p_star - p) / self.kl(p, p_star) if p != p_star else 0 for p in self.means]).sum()


class BanditBeta(Bandit):
    def __init__(self, parameter_list=None):
        if parameter_list is None:
            self.parameter_list = [(1, 1)]
        else:
            self.parameter_list = parameter_list
        self.arms = [arms.ArmFBeta(a, b) for a, b in parameter_list]
        self.n_arms = len(self.arms)

    def __repr__(self):
        return self.parameter_list__repr__()


class BanditExp(Bandit):
    def __init__(self, lambdas=None, n_arms=None):
        Bandit.__init__(self)
        if lambdas is None:
            if n_arms is None:
                self.n_arms = 2
            else:
                self.n_arms = n_arms
            self.lambdas = np.linspace(1, self.n_arms, self.n_arms)
        else:
            self.n_arms = len(lambdas)
        self.arms = [arms.ArmExp(lambd) for lambd in self.lambdas]

    def __repr__(self):
        return self.lambdas.__repr__()


class BanditFinite(Bandit):
    def __init__(self, parameter_list=None):
        if parameter_list is None:
            self.parameter_list = [(0, 1)]
        else:
            self.parameter_list = parameter_list
        self.arms = [arms.ArmFinite(x, p) for x, p in parameter_list]
        self.n_arms = len(self.arms)

    def __repr__(self):
        return self.parameter_list.__repr__()
