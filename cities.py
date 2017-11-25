import numpy as np
from scipy.spatial.distance import euclidean


class Cities:
    def __init__(self, n):
        # number of cities
        self.n = n

        # coordinates of cities
        self.c = np.random.randn(self.n, 2)

    # distances between cities
    @property
    def d(self):
        return np.array([[euclidean(c1, c2) for c2 in self.c] for c1 in self.c])
