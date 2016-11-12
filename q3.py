import numpy as np
import pandas as pd

from q2 import Policy


def value_iteration(tree, threshold=0.1):
    values = pd.Series(np.random.rand(len(tree.states)))
    new_values = pd.Series(np.zeros(len(tree.states)))
    error = np.inf
    policy = Policy(pd.Series([]))
    while error > threshold:
        q = pd.DataFrame(np.zeros((len(tree.states), tree.number_of_actions)))
        for state in tree.states:
            for a in [0, 1]:
                q.loc[state, a] = q(state, a, tree, values)
            new_values[state] = q.loc[state, :].max()
            policy.update(state, q.loc[state, :].argmax())
        error = (new_values - values).abs().max()
    return values, policy


# TODO Refactor bellman_operator using this
def q(state, action, tree, values):
    return tree.reward(state, action) + tree.gamma * pd.Series(
        [tree.dynamics[state, next_state, action] * values(next_state) for next_state in tree.states]).sum()
