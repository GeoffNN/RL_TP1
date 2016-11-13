import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Policy:
    def __init__(self, recommendations):
        self.recommendations = recommendations
        self.changed = True

    def recommend_for_state(self, x):
        return self.recommendations[x]

    def update(self, x, y):
        self.recommendations[x] = y

    def __repr__(self):
        return self.recommendations.__repr__()

    def __eq__(self, other):
        return self.recommendations == other.recommendations


def monte_carlo_estimator(policy, time_horizon, tree, number_of_samples=1000):
    values = pd.Series([])
    for state in tree.states:
        values[state] = monte_carlo_estimator_at_state(state, policy, time_horizon, tree, number_of_samples)
    return values


def monte_carlo_estimator_at_state(initial_state, policy, time_horizon, tree, number_of_samples=1000):
    values = pd.Series(
        [evaluate_reward(policy, time_horizon, initial_state, tree).sum() for k in range(number_of_samples)])
    return values.mean()


def monte_carlo_by_length(policy, time_horizon, tree, number_of_samples=1000):
    values = pd.DataFrame(np.zeros((len(tree.states), time_horizon)))
    for state in tree.states:
        values.loc[state] = monte_carlo_by_length_at_state(state, policy, time_horizon, tree, number_of_samples)
    return values


def monte_carlo_by_length_at_state(initial_state, policy, time_horizon, tree, number_of_samples=1000):
    values = pd.DataFrame(
        [evaluate_reward(policy, time_horizon, initial_state, tree).cumsum() for k in range(number_of_samples)])
    return values.mean()


def evaluate_reward(policy, time_horizon, initial_state, treeCut):
    state = initial_state
    rewards = pd.Series([])
    for t in range(time_horizon):
        cur_action = policy.recommend_for_state(state)
        state, cur_reward = treeCut.tree_sim(state, cur_action)
        rewards = rewards.append(pd.Series([(tree.gamma ** t) * cur_reward], index=[t]))
    return rewards


# def temporal_difference_estimator(policy, time_horizon, initial_state, treeCut, number_of_samples=1000):
#     state = initial_state
#     values = {0 for k in treeCut.states}
#     reward = 0
#     alpha = {state: 0 for state in treeCut.states}
#     for t in range(time_horizon):
#         cur_action = policy.recommend_for_state(state)
#         next_state, cur_reward = treeCut.tree_sim(state, cur_action)
#         reward += treeCut.gamma ** t * cur_reward
#         td = cur_reward + treeCut.gamma*values[next_state] - values[state]
#         values[state] = values[state] + td*alpha[state]


def bellman_operator(values, policy, tree):
    update_values = pd.Series([])
    for state in tree.states:
        update_values[state] = tree.reward[state, policy.recommend_for_state(state)] + tree.gamma * pd.Series(
            [tree.dynamics[state, arr_state, policy.recommend_for_state(state)] * values[arr_state] for arr_state in
             tree.states]).sum()
    return update_values


def bellman_estimator(policy, tree, thresh=0.01):
    values = pd.Series(np.random.rand(len(tree.states)))
    new_values = bellman_operator(values, policy, tree)
    while (new_values - values).abs().max() > thresh:
        values = new_values
        new_values = bellman_operator(values, policy, tree)
    return new_values

