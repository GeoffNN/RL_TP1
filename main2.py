from q21 import *

tree = TreeCut()

q_est, greedy_policies, rewards = q_learning(tree, epsilon=.1, n_episodes=200)
print("Q-learning done.")
values = pd.DataFrame([bellman_estimator(pol, tree) for pol in greedy_policies])
print("Values obtained for estimated policies")
optimal_value, opt_pol = value_iteration(tree)
opt_val = optimal_value[-1]

plt.figure()
(values - opt_val).abs().loc[:, 1].plot(title='Performance in the initial state')
plt.figure()
(values - opt_val).abs().max(axis=1).plot(title='Performance over all states')
plt.figure()
rewards.plot(title="Rewards by episode")
print("Done.")
