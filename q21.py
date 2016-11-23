from q1 import *
from q2 import *
from q3 import *


def alpha(x, a, i):
    return 1 / i


def q_learning(tree=TreeCut(), epsilon=.1, T_max=1000, n_episodes=1000):
    greedy_policies = []
    rewards = []
    for k in range(n_episodes):
        state = 1
        q_est = pd.DataFrame(
            np.zeros((len(tree.states), tree.number_of_actions)))
        cum_reward = 0
        for t in range(1, T_max):
            rand = bernoulli.rvs(epsilon)
            if rand:
                # randomize between No cut (0) and Cut (1)
                action = bernoulli.rvs(.5)
            else:
                action = q_est.loc[state, :].argmax()
            next_state, reward = tree.tree_sim(state, action)
            cum_reward += reward
            delta = reward + tree.gamma * q_est.loc[next_state, :].max() - q_est.loc[state, action]
            q_est.loc[state, action] += alpha(state, action, t)*delta
            state = next_state
        greedy_policies.append(Policy(pd.Series([q_est.loc[_,:].argmax() for _ in range(len(tree.states))], name=k)))
        rewards.append(cum_reward)
    rewards = pd.Series(rewards)
    return q_est, greedy_policies, rewards
