from sympy import KroneckerDelta
import numpy as np
import matplotlib.pyplot as plt


class Hopfield:
    def __init__(self, d):
        """Initialize the Hopfield network.

        Args:
        n: number of cities
        d: distances between cities
        """
        self.n = d.shape[0]
        self.d = d

        # weights between neurons
        self.w = np.zeros((self.n, self.n, self.n, self.n))
        for x in range(self.n):
            for i in range(self.n):
                for y in range(self.n):
                    for j in range(self.n):
                        self.w[x][i][y][j] = -2 * (KroneckerDelta(x, y) + KroneckerDelta(i, j) + d[x][y] * KroneckerDelta((i + 1) % self.n, j))

        # biases to neurons
        self.b = np.zeros((self.n, self.n))
        for x in range(self.n):
            for i in range(self.n):
                self.b[x][i] = 4

        # states of neurons
        self.s = np.random.choice([0, 1], (self.n, self.n))

    @property
    def e(self):
        """Return the energy of Hopfield network at the time.

        Returns:
        e: energy of the Hopfield network
        """
        e = 0
        for x in range(self.n):
            for i in range(self.n):
                for y in range(self.n):
                    for j in range(self.n):
                        e += - 0.5 * self.w[x][i][y][j] * self.s[x][i] * self.s[y][j]
                e += - self.b[x][i] * self.s[x][i]
        return e

    @property
    def e_tsp(self):
        """Return the energy for the Travelling Salesman Problem at the time.

        This energy is the same as the energy returned by the function e().

        Returns:
        energy for the Travelling Salesman Problem
        """
        e1 = np.sum((np.sum(self.s, axis=1) - 1) ** 2)
        e2 = np.sum((np.sum(self.s, axis=0) - 1) ** 2)
        e3 = 0
        for x in range(self.n):
            for i in range(self.n):
                for y in range(self.n):
                    e3 += self.d[x][y] * self.s[x][i] * self.s[y][(i + 1) % self.n]
        return e1 + e2 + e3 - 2 * self.n

    def update(self, x, i):
        """Update the state of selected neuron.

        Args:
        x: index of the neuron to be updated
        i: index of the neuron to be updated

        Returns:
        updated state of the selected neuron
        """
        u = np.sum(self.w[x][i] * self.s) + self.b[x][i]
        if u > 0:
            return 1
        else:
            return 0

    def is_minimum(self):
        """Check if the energy is the minimum.

        Returns:
        True if the energy is the minimum, otherwise False
        """
        for x in range(self.n):
            for i in range(self.n):
                if self.s[x][i] != self.update(x, i):
                    return False
        return True

    def go_to_minimum(self):
        """Repeat updating until the energy becomes the minimum."""
        self.e_array = np.array([self.e])
        while not self.is_minimum():
            x = np.random.randint(self.n)
            i = np.random.randint(self.n)
            self.s[x][i] = self.update(x, i)
            self.e_array = np.append(self.e_array, self.e)

    def plot_energy(self):
        """Plot the energy."""
        plt.plot(self.e_array)
        plt.show()
