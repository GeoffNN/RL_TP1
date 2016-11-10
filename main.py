from scipy.stats import poisson, bernoulli
import numpy as np

# parameters
growth_param = 5
replanting_cost = 100
linear_wood_value = 10
maintenance_cost = 3
max_height = 150
number_of_actions = 2
proba_of_dying = .1
death = 'death'
dead_index = 0
no_cut = 0
cut = 1
sappling_height = 1


def tree_sim(cur_state, action):
    if cur_state is death:
        if action is cut:
            next_state = sappling_height
            reward = -replanting_cost
        else:
            next_state = death
            reward = 0
    else:
        if action is cut:
            next_state = sappling_height
            reward = linear_wood_value * cur_state - replanting_cost
        else:
            death = bernoulli.rvs(proba_of_dying)
            if death:
                next_state = 'death'
                reward = -maintenance_cost
            else:
                growth = poisson.rvs(growth_param)
                if growth + cur_state > max_height:
                    next_state = max_height
                else:
                    next_state = cur_state + growth
                reward = -maintenance_cost
    return next_state, reward


def tree_MDP():
    dynamics = np.zeros((max_height + 1, max_height + 1, number_of_actions))
    rewards = np.zeros((max_height + 1, number_of_actions))

    dynamics[:, sappling_height, cut] = np.array([1] * max_height)
    dynamics[dead_index, dead_index, no_cut] = 1
    dynamics[max_height, max_height, no_cut] = 1 - proba_of_dying
    dynamics[1:, dead_index, no_cut] = proba_of_dying
    for cur_state in range(1, max_height):
        for next_state in range(1, max_height):
            dynamics[cur_state, next_state, no_cut] = (1 - proba_of_dying) * \
                                                      poisson.pmf(next_state - cur_state, growth_param)
        dynamics[cur_state, max_height, no_cut] = (1 - proba_of_dying) * (1 - poisson.cdf(cur_state, growth_param))


    return dynamics, rewards
