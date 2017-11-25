from cities import Cities
from hopfield import Hopfield


def main():
    n = 10
    c = Cities(n)
    h = Hopfield(c.d)
    h.go_to_minimum()
    h.plot_energy()


if __name__ == '__main__':
    main()
