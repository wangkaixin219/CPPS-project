import numpy as np


class RandomProcess(object):
    def __init__(self, size):
        self.size = size

    def sample(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class GaussianRandomProcess(RandomProcess):
    def __init__(self, size, mu=0., sigma=1.):
        super(GaussianRandomProcess, self).__init__(size=None)
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        sample_data = np.random.normal(self.mu, self.sigma, self.size)
        return sample_data

    def reset(self):
        pass


class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size, theta=.15, mu=0., sigma=1., dt=1e-2):
        super(OrnsteinUhlenbeckProcess, self).__init__(size)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.random.normal(self.mu, self.sigma, self.size)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(self.size)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.random.normal(self.mu, self.sigma, self.size)

