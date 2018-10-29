#!/usr/bin/env python3
from __future__ import annotations

from numpy.random import uniform, choice, shuffle
from numpy import ndarray, array, arange
from collections import Callable
from argparse import Namespace
from numba import jit, prange


class BaseParameters(object):
    def __init__(self, n: float, f: float, c: float, u: float, d: float, kappa: float, omega: float):
        """ Constructor. This is just meant to be a data class for the mutation model.

        :param n: Population size, used for determining the number of generations between events.
        :param f: Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
        :param c: Constant bias for the upward mutation rate.
        :param u: Linear bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        """
        self.n, self.f, self.c, self.u, self.d, self.kappa, self.omega = \
            round(n), f, c, u, d, round(kappa), round(omega)

        self.PARAMETER_COUNT = 7

    def __iter__(self):
        """ Return each our of parameters in the following order: n, f, c, u, d, kappa, omega

        :return: Iterator for all of our parameters.
        """
        for parameter in [self.n, self.f, self.c, self.u, self.d, self.kappa, self.omega]:
            yield parameter

    def __len__(self):
        """ The number of parameters that exist here.

        :return: The number of parameters we have.
        """
        return self.PARAMETER_COUNT

    @staticmethod
    def from_args(arguments: Namespace, is_sigma: bool = False) -> BaseParameters:
        """ Given a namespace, return a BaseParameters object with the appropriate parameters. If 'is_sigma' is
        toggled, we look for the sigma arguments in our namespace instead. This is commonly used with an ArgumentParser
        instance.

        :param arguments: Arguments from some namespace.
        :param is_sigma: If true, we search for 'n_sigma', 'f_sigma', ... Otherwise we search for 'n', 'f', ...
        :return: New BaseParameters object with the parsed in arguments.
        """
        if not is_sigma:
            return BaseParameters(n=arguments.n,
                                  f=arguments.f,
                                  c=arguments.c,
                                  u=arguments.u,
                                  d=arguments.d,
                                  kappa=arguments.kappa,
                                  omega=arguments.omega)
        else:
            return BaseParameters(n=arguments.n_sigma,
                                  f=arguments.f_sigma,
                                  c=arguments.c_sigma,
                                  u=arguments.u_sigma,
                                  d=arguments.d_sigma,
                                  kappa=arguments.kappa_sigma,
                                  omega=arguments.omega_sigma)

    @staticmethod
    def from_walk(theta: BaseParameters, pi_sigma: BaseParameters, walk: Callable) -> BaseParameters:
        """ Generate a new point from some walk function. We apply this walk function to each dimension, using the
        walking parameters specified in 'pi_sigma'. 'walk' must accept two variables, with the first being
        the point to walk from and second being the parameter to walk with. We must be within bounds.

        :param theta: Current point in our model space. The point we are currently walking from.
        :param pi_sigma: Walking parameters. These are commonly deviations.
        :param walk: For some point theta, generate a new one with the corresponding pi_sigma.
        :return: A new BaseParameters (point).
        """
        while True:
            theta_proposed = BaseParameters(n=walk(theta.n, pi_sigma.n),
                                            f=walk(theta.f, pi_sigma.f),
                                            c=walk(theta.c, pi_sigma.c),
                                            u=walk(theta.u, pi_sigma.u),
                                            d=walk(theta.d, pi_sigma.d),
                                            kappa=walk(theta.kappa, pi_sigma.kappa),
                                            omega=walk(theta.omega, pi_sigma.omega))

            if theta_proposed.n > 0 and \
                    theta_proposed.f >= 0 and \
                    theta_proposed.c > 0 and \
                    theta_proposed.u >= 1 and \
                    theta_proposed.d >= 0 and \
                    0 < theta_proposed.kappa < theta_proposed.omega:
                break

        return theta_proposed


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def triangle_n(a: int) -> int:
    """ Triangle number generator. Given 'a', return a choose 2. Optimized by Numba.

    :param a: Which triangle number to return.
    :return: The a'th triangle number.
    """
    return int(a * (a + 1) / 2)


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def mutate_n(i: int, c: float, u: float, d: float, kappa: int, omega: int) -> int:
    """ TODO: Finish documentation of the mutate_n method.

    :param i:
    :param c:
    :param u:
    :param d:
    :param kappa:
    :param omega:
    :return:
    """
    # If we reached some value kappa, we do not mutate.
    if i == kappa:
        return i

    # Compute our upward mutation rate. We are bounded by omega.
    i = min(omega, i + 1) if uniform(0, 1) < c + i * (d / u) else i

    # Compute our downward mutation rate. We are bounded by kappa.
    return max(kappa, i - 1) if uniform(0, 1) < i * d else i


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def coalesce_n(tau: int, coalescent_tree: ndarray) -> None:
    """ Generate the current coalescence event by sampling from our ancestors to our descendants. We save the indices
    of our ancestors to our descendants here (like a pointer), as opposed to their length.

    :param tau: TODO: Finish documentation for coalesce_n.
    :param coalescent_tree: Ancestor history chain, whose generations are indexed by triangle numbers.
    :return: None.
    """
    # We save the indices of our ancestors to our descendants.
    coalescent_tree[triangle_n(tau + 1) + 1:triangle_n(tau + 2)] = arange(triangle_n(tau), triangle_n(tau + 1))
    coalescent_tree[triangle_n(tau + 1)] = choice(coalescent_tree[triangle_n(tau + 1) + 1:triangle_n(tau + 2)])
    shuffle(coalescent_tree[triangle_n(tau + 1):triangle_n(tau + 2)])


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def evolve_n(coalescent_tree: ndarray, tau: int, n: int, f: float, c: float, u: float, d: float,
             kappa: int, omega: int) -> None:
    """ TODO: Finish documentation of the evolve_n method.

    Focal bias defined as \hat{L} = \frac{-c}{\frac{d}{u} - d}.

    :param coalescent_tree:
    :param tau:
    :param n:
    :param f:
    :param c:
    :param u:
    :param d:
    :param kappa:
    :param omega:
    :return:
    """
    # We define our ancestors and descendants for the tau'th coalescent.
    descendants = coalescent_tree[triangle_n(tau + 1):triangle_n(tau + 2)]

    # Iterate through each of the descendants (currently pointers) and determine each ancestor.
    for k in prange(descendants.size):
        descendant_to_evolve = coalescent_tree[descendants[k]]

        # Evolve each ancestor according to the average time to coalescence and the scaling factor f.
        for _ in range(max(1, round(f * 2 * n / triangle_n(tau + 1)))):
            descendant_to_evolve = mutate_n(descendant_to_evolve, c, u, d, kappa, omega)

        # Save our descendant state.
        descendants[k] = descendant_to_evolve


class Population(object):
    def __init__(self, theta: BaseParameters):
        """ Constructor. Perform the bound checking for each parameter here.

        :param theta: Parameter models to use to evolve our population.
        """
        from numpy import empty, nextafter

        # Ensure that all variables are bounded properly.
        theta.n = max(theta.n, 0)
        theta.f = max(theta.f, 0)
        theta.c = max(theta.c, nextafter(0, 1))  # Dr. Reed gave a strict bound > 0 here, but I forget why...
        theta.u = max(theta.u, 1.0)
        theta.d = max(theta.d, 0)
        theta.kappa = max(theta.kappa, 0)
        theta.omega = max(theta.omega, theta.kappa)
        self.theta = theta

        # Define our ancestor chain.
        self.coalescent_tree = empty([triangle_n(2 * self.theta.n)], dtype='int')
        self.offset = 0

        # Trace our tree. We do not perform repeat length determination at this step.
        self._trace_tree()
        self.is_evolved = False

    def _trace_tree(self):
        """ TODO: Finish documentation for _trace_tree method.

        :return:
        """
        # Generate 2N - 1 coalescence events.
        [coalesce_n(tau, self.coalescent_tree) for tau in range(0, 2 * self.theta.n - 1)]

    def evolve(self, i_0: ndarray) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until '2N' alleles are present.

        :param i_0 Array of starting common ancestors.
        :return: The evolved generation from the common ancestor.
        """
        # If we have evolved, then return what we currently have. Otherwise, raise our flag.
        if self.is_evolved:
            return self.coalescent_tree[-2 * self.theta.n:]
        else:
            self.is_evolved = True

        # Determine our offset, and seed our ancestors for the tree.
        self.offset = i_0.size - 1
        self.coalescent_tree[triangle_n(self.offset):triangle_n(self.offset + 1)] = i_0

        # From our common ancestors, descend forward in time and populate our tree with repeat lengths.
        [evolve_n(self.coalescent_tree, tau, self.theta.n, self.theta.f, self.theta.c, self.theta.u, self.theta.d,
                  self.theta.kappa, self.theta.omega) for tau in range(self.offset, 2 * self.theta.n - 1)]

        # Return the evolved generation of ancestors.
        return self.coalescent_tree[-2 * self.theta.n:]


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentParser
    from matplotlib import pyplot as plt

    parser = ArgumentParser(description='Simulate the evolution of single population.')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)
    paa('-image', 'Image file to save resulting repeat length distribution (histogram) to.', str)

    parser.add_argument('-i_0', help='Repeat lengths of starting ancestors.', type=int, nargs='+')
    paa('-n', 'Starting population size.', int)
    paa('-f', 'Scaling factor for total mutation rate.', float)
    paa('-c', 'Constant bias for the upward mutation rate.', float)
    paa('-u', 'Linear bias for the upward mutation rate.', float)
    paa('-d', 'Linear bias for the downward mutation rate.', float)
    paa('-kappa', 'Lower bound of repeat lengths.', int)
    paa('-omega', 'Upper bound of repeat lengths.', int)
    main_arguments = parser.parse_args()  # Parse our arguments.

    # Evolve some population.
    main_population = Population(BaseParameters.from_args(main_arguments))
    main_descendants = main_population.evolve(array(main_arguments.i_0))

    # Display a histogram.
    plt.hist(main_descendants, bins=range(min(main_descendants), max(main_descendants)))
    plt.savefig(main_arguments.image) if main_arguments.image is not None else plt.show()
