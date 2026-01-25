import numpy as np

OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter

class OrnsteinUhlenbeckProcess:
    """Ornstein-Uhlenbeck process for exploration in DDPG.
    
    Ref: Uhlenbeck & Ornstein (1930), used in Lillicrap et al. (2015).
    """
    def __init__(self, size, seed, sigma=0.2, mu=0.0):
        """
        Params
        ======
            size (int): dimension of the noise (usually same as action space)
            seed (int): random seed
            mu (float): long-running mean
            theta (float): speed of mean reversion
            sigma (float): scale of random fluctuations
        """
        self.mu = mu * np.ones(size)
        self.theta = OU_THETA
        self.sigma = OU_SIGMA
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state