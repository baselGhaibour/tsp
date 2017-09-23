import numpy as np


class Cities:
    def __init__(self, n):
        # number of cities
        self.n = n

        # coordinates of cities
        self.c = np.random.randn(self.n, 2)

    # distances between cities
    @property
    def d(self):
        return np.array([[np.sum((self.c[x] - self.c[y]) ** 2) for y in range(self.n)] for x in range(self.n)])
