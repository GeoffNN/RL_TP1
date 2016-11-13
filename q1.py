from scipy.stats import poisson, bernoulli, randint
import numpy as np


class TreeCut:
    def __init__(self, growth_param=5, replanting_cost=100, linear_wood_value=10,
                 maintenance_cost=3, max_height=15, proba_of_dying=.1,
                 sappling_height=1, gamma=1./(1+0.05)):

        self.growth_param = growth_param
        self.replanting_cost = replanting_cost
        self.linear_wood_value = linear_wood_value
        self.maintenance_cost = maintenance_cost
        self.max_height = max_height
        self.proba_of_dying = proba_of_dying
        self.sappling_height = sappling_height
        self.gamma = gamma

        self.states = range(self.max_height + 1)
        self.number_of_actions = 2
        self.death = 0
        self.dead_index = 0
        self.no_cut = 0
        self.cut = 1

        self.dynamics, self.reward = self.tree_MDP()

    def tree_sim(self, cur_state, action):
        if cur_state is self.death:
            if action is self.cut:
                next_state = self.sappling_height
                reward = -self.replanting_cost
            else:
                next_state = self.death
                reward = 0
        else:
            if action is self.cut:
                next_state = self.sappling_height
                reward = self.linear_wood_value * cur_state - self.replanting_cost
            else:
                tree_is_dying = bernoulli.rvs(self.proba_of_dying)
                if tree_is_dying:
                    next_state = self.death
                    reward = -self.maintenance_cost
                else:
                    next_state = randint.rvs(cur_state, self.max_height + 1)
                    reward = -self.maintenance_cost
        return next_state, reward

    def tree_MDP(self):
        dynamics = np.zeros((self.max_height + 1, self.max_height + 1, self.number_of_actions))
        rewards = np.zeros((self.max_height + 1, self.number_of_actions))

        dynamics[:, self.sappling_height, self.cut] = np.array([1] * (self.max_height + 1))
        dynamics[self.dead_index, self.dead_index, self.no_cut] = 1
        dynamics[self.max_height, self.max_height, self.no_cut] = 1 - self.proba_of_dying
        dynamics[1:, self.dead_index, self.no_cut] = self.proba_of_dying
        for cur_state in range(1, self.max_height):
            for next_state in range(cur_state, self.max_height + 1):
                dynamics[cur_state, next_state, self.no_cut] = (1 - self.proba_of_dying) * 1. / (
                    self.max_height - cur_state + 1)

        rewards[self.dead_index, :] = [0, -self.replanting_cost]
        rewards[1:, self.no_cut] = [-self.maintenance_cost for k in range(self.max_height)]
        rewards[1:, self.cut] = [self.linear_wood_value * cur_state - self.replanting_cost for cur_state in
                                 range(1, self.max_height + 1)]

        return dynamics, rewards
