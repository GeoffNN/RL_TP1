"""Answer to question 1. of TD2."""

from q1 import *
from q2 import *
from q3 import *


def alpha(x, a, i):
    if i == 0:
        return 0
    else:
        return 1. / i


def q_learning(tree=TreeCut(), epsilon=.1, T_max=1000, n_episodes=100):
    greedy_policies = []
    rewards = []
    q_est = pd.DataFrame(
        np.zeros((len(tree.states), tree.number_of_actions)))
    for k in range(n_episodes):
        n_visits = pd.DataFrame(np.zeros((len(tree.states), tree.number_of_actions)))
        state = 1
        cum_reward = 0
        for t in range(1, T_max):
            rand = bernoulli.rvs(epsilon)
            if rand:
                # randomize between No cut (0) and Cut (1)
                action = bernoulli.rvs(0.5)
            else:
                action = q_est.loc[state, :].argmax()
            next_state, reward = tree.tree_sim(state, action)
            cum_reward += reward
            delta = reward + tree.gamma * q_est.loc[next_state, :].max() - q_est.loc[state, action]
            q_est.loc[state, action] += alpha(state, action, n_visits.loc[state, action]) * delta
            n_visits.loc[state, action] += 1
            state = next_state
        greedy_policies.append(Policy(pd.Series([q_est.loc[_, :].argmax() for _ in range(len(tree.states))], name=k)))
        rewards.append(cum_reward)
    rewards = pd.Series(rewards)
    return q_est, greedy_policies, rewards
