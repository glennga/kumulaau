#!/usr/bin/env python3
from __future__ import annotations

from numpy import ndarray, asarray, sqrt
from typing import Iterable, Callable
from argparse import Namespace
import pop


class Model(object):
    @staticmethod
    def simulator(func: Callable) -> Callable:
        """ Decorator for all simulator functions. All we do here is verify that (a) theta is an argument, (b) i_0
        is an argument, and (c) that i_0 is a triangle number.

        :param func: Simulator function to wrap.
        :return: Simulator function with appropriate argument and bounds checking.
        """
        def simulator_wrapper(*args, **kwargs):
            if 'i_0' not in kwargs:  # Verify that we are issued the bare minimum correct arguments.
                raise KeyError("i_0 not passed as an argument.")
            elif 'theta' not in kwargs:
                raise KeyError("theta not passed as an argument.")

            # Verify that number of seed lengths passed is a triangle number.
            if 0.5 * (sqrt(8 * len(kwargs['i_0']) + 1) - 1) % 1 != 0:
                raise ValueError("i_0 is not a triangle number.")

            return func(*args, **kwargs)

        return simulator_wrapper

    @staticmethod
    def trace(n, f, c, d, kappa, omega):
        """ A wrapper for the pop module trace method. This returns a C pointer that holds the topology and population
        parameters from the trace method.

        :param n: Population size, used for determining the number of generations between events.
        :param f: Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        :return: Pointer to a pop module C structure (tree).
        """
        return pop.trace(n, f, c, d, kappa, omega)

    @staticmethod
    def evolve(p, i_0: Iterable) -> ndarray:
        """ A wrapper for the pop module evolve method. Given the C pointer from a trace call and initial lengths,
        we resolve our repeat lengths and return our result as a numpy array.

        :param p: Pointer to a pop module C structure (tree)
        :param i_0: Array of starting lengths.
        :return: Array of repeat lengths.
        """
        return asarray(pop.evolve(p, [i for i in i_0]))


def get_arguments() -> Namespace:
    """ Create the CLI and parse the arguments, if used as our main script.

    :return: Namespace of all values.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Simulate the evolution of single population.')
    list(map(lambda a: parser.add_argument(a[0], help=a[1], type=a[2], nargs=a[3]), [
        ['-image', 'Image file to save resulting repeat length distribution (histogram) to.', str, None],
        ['-i_0', 'Repeat lengths of starting ancestors.', int, '+'],
        ['-n', 'Starting population size.', int, None],
        ['-f', 'Scaling factor for total mutation rate.', float, None],
        ['-c', 'Constant bias for the upward mutation rate.', float, None],
        ['-d', 'Linear bias for the downward mutation rate.', float, None],
        ['-kappa', 'Lower bound of repeat lengths.', int, None],
        ['-omega', 'Upper bound of repeat lengths.', int, None]
    ]))

    return parser.parse_args()


if __name__ == '__main__':
    from timeit import default_timer as timer
    from numpy import std, average
    from matplotlib import use

    arguments = get_arguments()  # Parse our arguments.

    use('TkAgg')  # Use a different backend (plotting to display).
    from matplotlib import pyplot as plt

    # Evolve some population 1000 times.
    start_t = timer()
    main_descendants = [pop.evolve(pop.trace(arguments.n,
                                             arguments.f,
                                             arguments.c,
                                             arguments.d,
                                             arguments.kappa,
                                             arguments.omega),
                                   arguments.i_0) for _ in range(1000)]
    end_t = timer()
    print('Time Elapsed (10000x): [\n\t' + str(end_t - start_t) + '\n]')

    # Determine the variance and mean of the generated populations.
    print('Mean (10000x): [\n\t' + str(average(main_descendants)) + '\n]')
    print('Deviation (10000x): [\n\t' + str(std(main_descendants)) + '\n]')

    # Display a histogram of the first generated populace.
    plt.hist(main_descendants[0], bins=range(min(main_descendants[0]), max(main_descendants[0])))
    plt.savefig(arguments.image) if arguments.image is not None else plt.show()
