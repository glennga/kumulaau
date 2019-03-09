#!/usr/bin/env python3
from __future__ import annotations

from numpy import ndarray, array, asarray
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List
import pop


class Population(ABC):
    @abstractmethod
    def _transform_bounded(self, theta):
        """ Return a new parameter set, bounded by some constraints.

        :param theta: Parameter set to transform.
        :return: A new parameter set, properly bounded.
        """
        raise NotImplementedError

    @abstractmethod
    def _trace_trees(self, theta) -> List:
        """ Generate a list of pointers to trees generated by the pop module.

        :param theta: Parameter set to use with tree tracing.
        :return: List of pointers to C structures in pop module, order maintained by child class.
        """
        raise NotImplementedError

    @abstractmethod
    def _resolve_lengths(self, i_0: ndarray) -> ndarray:
        """ Generate a list of lengths, whose topology is determined by the child class.

        :return: List of repeat lengths.
        """
        raise NotImplementedError

    def __init__(self, theta):
        """ Constructor. Here bound our parameter set theta and build our trees.

        :param theta: A parameter set for the given population model.
        """
        self.is_evolved, self.evolved = False, array([])  # Our default values.
        self.theta = theta  # Save our parameter set.

        # Trace our trees. The result should return pointers to one or more C structures.
        self.tree_pointers = self._trace_trees(self._transform_bounded(theta))

    def evolve(self, i_0: ndarray) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until '2n' alleles are present.

        :param i_0 Array of starting common ancestors.
        :return: The evolved generation from the common ancestor.
        """
        if self.is_evolved:  # If we have evolved, then return what we currently have. Otherwise, raise our flag.
            return self.evolved
        else:
            self.is_evolved = True

        # Evolve our population using the tree(s) generated at instantiation.
        self.evolved = self._resolve_lengths(i_0)

        # Return the evolved generation of ancestors.
        return self.evolved

    @staticmethod
    def _pop_trace(n, f, c, d, kappa, omega):
        """ A wrapper for the pop module trace method.

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
    def _pop_evolve(p, i_0: ndarray) -> ndarray:
        """ A wrapper for the pop module evolve method.

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

    main_arguments = get_arguments()  # Parse our arguments.

    use('TkAgg')  # Use a different backend (plotting to display).
    from matplotlib import pyplot as plt

    # Evolve some population 1000 times.
    start_t = timer()
    main_descendants = [pop.evolve(pop.trace(main_arguments.n, main_arguments.f, main_arguments.c, main_arguments.d,
                                             main_arguments.kappa, main_arguments.omega), main_arguments.i_0)
                        for _ in range(1000)]
    end_t = timer()
    print('Time Elapsed (10000x): [\n\t' + str(end_t - start_t) + '\n]')

    # Determine the variance and mean of the generated populations.
    print('Mean (10000x): [\n\t' + str(average(main_descendants)) + '\n]')
    print('Deviation (10000x): [\n\t' + str(std(main_descendants)) + '\n]')

    # Display a histogram of the first generated populace.
    plt.hist(main_descendants[0], bins=range(min(main_descendants[0]), max(main_descendants[0])))
    plt.savefig(main_arguments.image) if main_arguments.image is not None else plt.show()
