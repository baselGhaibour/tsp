from cities import Cities
from hopfield import Hopfield


def main():
    n = 4
    hyperparameter = 0.1
    c = Cities(n)
    h = Hopfield(c.c, hyperparameter)
    #h.go_to_minimum()
    #h.plot_energy()
    h.show_shortest_route()


if __name__ == '__main__':
    main()
