import numpy as np
import copy

from q2 import Policy, bellman_estimator


def value_iteration(tree, threshold=0.1):
    values = []
    values.append(np.random.rand(len(tree.states)))
    new_values = np.zeros(len(tree.states))
    error = np.inf
    policy = Policy([0 for _ in range(len(tree.states))])
    count = 0
    while error > threshold:
        q = q_fun(tree, values[count])
        values.append(copy.deepcopy(new_values))
        count += 1
        for state in tree.states:
            new_values[state] = q[state, :].max()
            policy.update(state, q[state, :].argmax())
        error = (new_values - values[count]).abs().max()
    return values, policy


def policy_iteration(tree, n_iterations=100):
    policy = Policy([0 for _ in range(len(tree.states))])
    values = []
    for k in range(n_iterations):
        values.append(bellman_estimator(policy, tree, thresh=0.01))
        q = q_fun(tree, values[k])
        old_policy = copy.deepcopy(policy)
        for x in tree.states:
            policy.update(x, q[x, :].argmax())
        if policy == old_policy:
            break
    return values, policy


def q_fun(tree, values):
    q = np.zeros((len(tree.states), tree.number_of_actions))
    for state in tree.states:
        for a in [0, 1]:
            q[state, a] = q_fun_values(state, a, tree, values)
    return q


# TODO Refactor bellman_operator using this
def q_fun_values(state, action, tree, values):
    return tree.reward[state, action] + tree.gamma * np.array(
        [tree.dynamics[state, next_state, action] * values[next_state] for next_state in tree.states]).sum()
