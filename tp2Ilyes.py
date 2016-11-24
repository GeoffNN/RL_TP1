import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
# import seaborn as sns
import time


class ArmBernoulli():
    """Bernoulli arm"""

    def __init__(self, p):
        """
        p: Bernoulli parameter
        """
        self.p = p
        self.mean = p
        self.var = p * (1 - p)

    def sample(self):
        reward = (np.random.random() < self.p) * 1
        return reward


class ArmBeta():
    """arm having a Beta distribution"""

    def __init__(self, a, b):
        """
        a: first beta parameter
        b: second beta parameter
        """
        self.a = a
        self.b = b
        self.mean = a / (a + b)
        self.var = (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self):
        reward = np.random.beta(self.a, self.b)
        return reward


class ArmExp():
    """arm with trucated exponential distribution"""

    def __init__(self, lambd):
        """
        lambd: parameter of the exponential distribution
        """
        self.lambd = lambd
        self.mean = (1 / lambd) * (1 - np.exp(-lambd))
        self.var = 1  # compute it yourself!

    def sample(self):
        reward = min(-1 / self.lambd * np.log(np.random.random()), 1)
        return reward


def simu(p):
    """
    draw a sample of a finite-supported distribution that takes value
    k with porbability p(k)
    p: a vector of probabilities
    """
    q = p.cumsum()
    u = np.random.random()
    i = 0
    while u > q[i]:
        i += 1
        if i >= len(q):
            raise ValueError("p does not sum to 1")
    return i


class ArmFinite():
    """arm with finite support"""

    def __init__(self, X, P):
        """
        X: support of the distribution
        P: associated probabilities
        """
        self.X = np.array(X)
        self.P = np.array(P)
        self.mean = (self.X * self.P).sum()
        self.var = (self.X ** 2 * self.P).sum() - self.mean ** 2

    def sample(self):
        i = simu(self.P)
        reward = self.X[i]
        return reward


class MultiArmBandit:
    """a multi arm bandit"""

    def __init__(self, arms):
        """
        :param arms: list of arms
        :return: a MultiArmBandit object
        """
        self.arms = arms
        self.N = len(arms)
        self.means = [arm.mean for arm in arms]
        self.mu_max = np.max(self.means)

    def display_means(self):
        print(self.means)


def UCB1(mab, T=5000):
    """
    UCB1 algorithms
    :param mab: a MultiArmBandit object
    :param T: number of iterations
    :return: rew, draws : rewards and draws at each step until T
    """
    # init
    K = mab.N  # number of arms
    arms = mab.arms
    rew = np.zeros(T)
    draws = np.zeros(T)
    N = np.zeros(K)
    S = np.zeros(K)

    for t in range(K):
        reward = arms[t].sample()
        N[t] += 1
        S[t] += reward
        rew[t] = reward
        draws[t] = t

    for t in range(K, T):
        A = np.argmax(S / N + np.sqrt(np.log(t) / (2 * N)))
        reward = arms[A].sample()
        N[A] += 1
        S[A] += reward
        rew[t] = reward
        draws[t] = A

    # rint(S / N)
    return rew, draws


def TS(mab, T=5000):
    """
    Thompson sampling algorithm
    :param mab: a MultiArmBandit object
    :param T: number of iterations
    :return: rew, draws : rewards and draws at each step until T
    """
    K = mab.N  # number of arms
    arms = mab.arms
    rew = np.zeros(T)
    draws = np.zeros(T)
    N = np.zeros(K)
    S = np.zeros(K)

    for t in range(T):
        A = np.argmax(np.array([np.random.beta(S[a] + 1, N[a] - S[a] + 1) for a in range(K)]))
        reward = arms[A].sample()
        N[A] += 1
        S[A] += reward
        rew[t] = reward
        draws[t] = A

    # print(S / N)
    return rew, draws


def naif(mab, T=5000):
    """

    :param mab:
    :param T:
    :return:
    """
    K = mab.N  # number of arms
    arms = mab.arms
    rew = np.zeros(T)
    draws = np.zeros(T)
    N = np.zeros(K)
    S = np.zeros(K)

    for t in range(K):
        reward = arms[t].sample()
        N[t] += 1
        S[t] += reward
        rew[t] = reward
        draws[t] = t

    for t in range(T):
        A = np.argmax(S / N)
        reward = arms[A].sample()
        N[A] += 1
        S[A] += reward
        rew[t] = reward
        draws[t] = A

    # print(S / N)
    return rew, draws


def regret_estimator(method, mab, nb_simul=100, T=5000):
    cumsum = np.zeros((nb_simul, T))
    for n in range(nb_simul):
        rew, _ = method(mab, T)
        cumsum[n] = np.cumsum(rew)
    return np.arange(T) * mab.mu_max - np.mean(cumsum, axis=0)


def kl(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def C(p):
    """
    returns the lay and robbing lower bound
    :param p: array of a MultiArmBandit means
    :return: C(p)
    """
    p_star = np.max(p)
    return np.sum([(p_star - p_a) / kl(p_a, p_star) for p_a in p if p_a != p_star])


def general_TS(mab, T=5000):
    """
    Thompson sampling algorithm for general stochastic MABs
    For this, we reuse the same formulation as for binary MABs, but by adding a rejection sampling,
    to adapt it to the arms distribution.
    This algorithm works also for binary MABs
    :param mab: a general MultiArmBandit object
    :param T: number of iterations
    :return: rew, draws : rewards and draws at each step until T
    """
    K = mab.N  # number of arms
    arms = mab.arms
    rew = np.zeros(T)
    draws = np.zeros(T)
    N = np.zeros(K)
    S = np.zeros(K)

    for t in range(T):
        A = np.argmax(np.array([np.random.beta(S[a] + 1, N[a] - S[a] + 1) for a in range(K)]))
        reward = arms[A].sample()
        N[A] += 1
        S[A] += bernoulli.rvs(reward)
        rew[t] = reward
        draws[t] = A

    # print(S / N)
    return rew, draws


def plot_regret(mab, nb_simul=100, T=5000, plot_naif=False):
    """
    plot regrets for UCB1, general_TS, and naif if wanted.
    :param mab: a general MAB object
    :param nb_simul: number of simulations
    :param T: horizon
    :return: None, plots the regret curves
    """

    print('means :', mab.means)

    reg_UCB1 = regret_estimator(UCB1, mab, nb_simul=nb_simul, T=T)
    # reg_TS_gen = regret_estimator(general_TS, mab, nb_simul=nb_simul, T=T)
    reg_TS = regret_estimator(TS, mab, nb_simul=nb_simul, T=T)
    reg_naif = regret_estimator(naif, mab, nb_simul=nb_simul, T=T)

    # Oracle with Lai and Robbins lower bound
    oracle = C(mab.means) * np.log(np.arange(1, T))

    fig = plt.figure()
    plt.plot(reg_UCB1, label='UCB1')
    plt.plot(reg_TS, label='TS')
    # plt.plot(reg_TS_gen, label='general TS')
    if plot_naif:
        plt.plot(reg_naif, label='naif')
    plt.plot(oracle, label='oracle')
    plt.legend(loc=4)
    # sns.despine(fig)
    plt.show()
    return fig


start = time.time()

Arm1 = ArmBernoulli(0.5)
Arm2 = ArmBernoulli(0.3)
Arm3 = ArmBernoulli(0.25)
Arm4 = ArmBernoulli(0.1)
mab1 = MultiArmBandit([Arm1, Arm2, Arm3, Arm4])
# plot_regret(mab1)



mab2 = MultiArmBandit([ArmBernoulli(0.9), ArmBernoulli(0.1)])
# plot_regret(mab2)
start = time.time()
plot_regret(MultiArmBandit([ArmBeta(0.5, 0.5), ArmExp(4), ArmBernoulli(0.4), ArmExp(5)]), nb_simul=100, T=1000)
print(time.time() - start)