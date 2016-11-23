class Bernoulli:


    def __init__(self, n_arms=10, means = None):
        self.n_arms = n_arms
        if means is None:
            self.means = [1/n_arms for _ in range(n_arms)]
        else:
            if len(means) != n_arms:
                print("Please provide as many means as arms in the bandit.")
                quit()
            else:
                self.means= means