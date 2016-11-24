from q21 import *
import time
tree = TreeCut()

start = time.time()
q_est, greedy_policies, rewards = q_learning(tree, epsilon=.1, n_episodes=200)
print(time.time() - start)
print("Q-learning done.")
values = pd.DataFrame([bellman_estimator(pol, tree) for pol in greedy_policies])
print("Bellman done")
print("Values obtained for estimated policies")
start= time.time()
optimal_value, opt_pol = value_iteration(tree)
print(time.time()-start)
opt_val = optimal_value[-1]

plt.figure()
plt.plot((values - opt_val).abs().loc[:, 1])
plt.title('Performance in the initial state')
plt.figure()
plt.plot((values - opt_val).abs().max(axis=1))
plt.title('Performance over all states')
plt.figure()
plt.plot(rewards)
plt.title("Rewards by episode")
print("Done.")
