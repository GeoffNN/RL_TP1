from q1 import *
from q2 import *
from q3 import *


def q2_test():
    tree = TreeCut(max_height=20)
    basic_policy = Policy([0 if k < .9 * len(tree.states) else 1 for k in range(len(tree.states))])
    # mc_vals = monte_carlo_estimator(basic_policy, 250, tree)
    b_vals = bellman_estimator(basic_policy, tree, 0.01)
    mc_vals_for_plot = monte_carlo_by_length_at_state(1, basic_policy, 1000, tree)
    (mc_vals_for_plot - b_vals[1]).plot(label="Difference between Monte Carlo and Bellman estimators")
    return b_vals, mc_vals_for_plot

def q3_pi_test():
    tree = TreeCut(max_height=100)
    return policy_iteration(tree)

def q3_vi_test():
    tree = TreeCut(max_height=100)
    return value_iteration(tree)