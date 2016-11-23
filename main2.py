from q21 import *

tree = TreeCut()

q_est, greedy_policies, rewards = q_learning(tree, n_episodes=1000)

values = pd.DataFrame([bellman_estimator(pol, tree) for pol in greedy_policies])
optimal_value, _ = value_iteration(tree)

