#!/usr/bin/env python3
from __future__ import annotations

from numpy import ndarray, array, arange, empty, nextafter
from numpy.random import uniform, choice, shuffle, exponential
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


class Population(object):
    def __init__(self, theta: BaseParameters):
        """ Constructor. Perform the bound checking for each parameter here.

        :param theta: Parameter models to use to evolve our population.
        """
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
        self.coalescent_tree = empty([self.triangle(2 * self.theta.n)], dtype='int')
        self.offset = 0

        # Trace our tree. We do not perform repeat length determination at this step.
        self._trace_tree(self.coalescent_tree, self.theta.n, self._coalesce, self.triangle)
        self.is_evolved = False

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu')
    def triangle(a: int) -> int:
        """ Triangle number generator. Given 'a', return a choose 2. Optimized by Numba.

        :param a: Which triangle number to return.
        :return: The a'th triangle number.
        """
        return int(a * (a + 1) / 2)

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu')
    def _mutate(ell: int, c: float, u: float, d: float, kappa: int, omega: int) -> int:
        """ Given a repeat length 'ell', we mutate this repeat length up or down dependent on our parameters
        c (upward constant bias), u (upward linear bias), and d (downward linear bias). If we reach our lower bound
        kappa, we do not mutate further. Optimized by Numba. The mutation model gives us the following focal bias:

        \hat{L} = \frac{-c}{\frac{d}{u} - d}.

        :param ell: The current repeat length to mutate.
        :param c: Constant bias for the upward mutation rate.
        :param u: Linear bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of our repeat length space. If 'ell = kappa', we do not mutate.
        :param omega: Upper bound of our repeat length space.
        :return: A mutated repeat length, either +1, -1, or +0 from the given 'ell'.
        """
        # If we reached some value kappa, we do not mutate.
        if ell == kappa:
            return ell

        # Compute our upward mutation rate. We are bounded by omega.
        ell = min(omega, ell + 1) if uniform(0, 1) < c + ell * (d / u) else ell

        # Compute our downward mutation rate. We are bounded by kappa.
        return max(kappa, ell - 1) if uniform(0, 1) < ell * d else ell

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _coalesce(tau: int, coalescent_tree: ndarray, triangle: Callable) -> None:
        """ Generate the current coalescence event by sampling from our ancestors to our descendants. We save the
        indices of our ancestors to our descendants here (like a pointer), as opposed to their length. Optimized by
        Numba.

        :param tau: The current coalescence event to generate (how far along our ancestor chain to generate).
        :param coalescent_tree: Ancestor history chain, whose generations are indexed by triangle numbers.
        :return: None.
        """
        # We save the indices of our ancestors to our descendants.
        coalescent_tree[triangle(tau + 1) + 1:triangle(tau + 2)] = arange(triangle(tau), triangle(tau + 1))
        coalescent_tree[triangle(tau + 1)] = choice(coalescent_tree[triangle(tau + 1) + 1:triangle(tau + 2)])
        shuffle(coalescent_tree[triangle(tau + 1):triangle(tau + 2)])

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _evolve(coalescent_tree: ndarray, tau: int, theta: ndarray, triangle: Callable, mutate: Callable) -> None:
        """ Given a coalescent tree whose values represent the indexes of their ancestor, perform repeat length
        determination for the tau'th event and generate a repeat length population. This must be perform in sequential
        order. Optimized by Numba.

        :param coalescent_tree: A 1D array containing the tree. We output the lengths to the next generation here.
        :param tau: The numbered coalescent event to perform repeat length determination for.
        :param theta: Our parameters, in the order of: n, f, c, u, d, kappa, omega.
        :param triangle: The triangle (binomial of two) function to use.
        :param mutate: The mutation function to use.
        :return: None.
        """
        n, f, c, u, d, kappa, omega = theta  # Unpack our parameters.

        # We define our descendants for the tau'th coalescent.
        descendants = coalescent_tree[triangle(tau + 1):triangle(tau + 2)]

        # Determine the time to coalescence. This is exponentially distributed, but the mean stays the same. Scale by f.
        t_coalescence = max(1, round(exponential(f * 2 * n / triangle(tau + 1))))

        # Iterate through each of the descendants (currently pointers) and determine each ancestor.
        for k in prange(descendants.size):
            descendant_to_evolve = coalescent_tree[descendants[k]]

            # Evolve each ancestor according to the average time to coalescence and the scaling factor f.
            for _ in range(t_coalescence):
                descendant_to_evolve = mutate(descendant_to_evolve, c, u, d, kappa, omega)

            # Save our descendant state.
            descendants[k] = descendant_to_evolve

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=False)
    def _trace_tree(coalescent_tree: ndarray, n: int, coalesce: Callable, triangle: Callable) -> None:
        """ Create a random evolutionary tree. The result is a 1D array, indexed by triangle(n). Repeat length
        determination does not occur at this stage, rather we determine which ancestor belongs to who. Optimized by
        Numba.

        :param coalescent_tree: A 1D array containing the tree to save to.
        :param n: The number of individuals to generate (our sample size, or the number of leaves we have).
        :param coalesce: The coalescence function to use to construct our tree.
        :param triangle: The triangle (binomial of two) function to use.
        :return: None.
        """
        for tau in range(0, 2 * n - 1):  # Generate 2N - 1 coalescence events.
            coalesce(tau, coalescent_tree, triangle)

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=False)
    def _evolve_n(coalescent_tree: ndarray, theta: ndarray, offset: int, evolve: Callable,
                  triangle: Callable, mutate: Callable) -> None:
        """ Perform repeat length determination for the entire coalescent tree. This overwrites the tree given to us,
        meaning that we can only do this once. The common ancestors for this array should be manually set before
        using this function. Optimized by Numba.

        :param coalescent_tree: A 1D array containing the tree we save our repeat lengths to.
        :param theta: Our parameters, in the order of: n, f, c, u, d, kappa, omega.
        :param offset: The number of common ancestors located in our tree. Determines where to start.
        :param evolve: Function to determine the repeat lengths of the tau'th coalescent event.
        :param triangle: The triangle (binomial of two) function to use.
        :param mutate: The mutation function to use.
        :return: None.
        """
        n, f, c, u, d, kappa, omega = theta  # Unpack our parameters.

        for tau in range(offset, 2 * n - 1):
            evolve(coalescent_tree, tau, theta, triangle, mutate)

    def evolve(self, i_0: ndarray) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until '2n' alleles are present.

        :param i_0 Array of starting common ancestors.
        :return: The evolved generation from the common ancestor.
        """
        if self.is_evolved:  # If we have evolved, then return what we currently have. Otherwise, raise our flag.
            return self.coalescent_tree[-2 * self.theta.n:]
        else:
            self.is_evolved = True

        # Determine our offset, and seed our ancestors for the tree.
        self.offset = i_0.size - 1
        self.coalescent_tree[self.triangle(self.offset):self.triangle(self.offset + 1)] = i_0

        # From our common ancestors, descend forward in time and populate our tree with repeat lengths.
        k = (self._evolve, self.triangle, self._mutate)
        self._evolve_n(self.coalescent_tree, array(list(self.theta)), self.offset, *k)

        # Return the evolved generation of ancestors.
        return self.coalescent_tree[-2 * self.theta.n:]


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentParser
    from matplotlib import pyplot as plt
    from timeit import default_timer as timer

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

    # Evolve some population 500 times.
    start_t = timer()
    [Population(BaseParameters.from_args(main_arguments)).evolve(array(main_arguments.i_0)) for _ in range(500)]
    end_t = timer()
    print('Time Elapsed (500x): [\n\t' + str(end_t - start_t) + '\n]')

    main_population = Population(BaseParameters.from_args(main_arguments))
    main_descendants = main_population.evolve(array(main_arguments.i_0))

    # Display a histogram.
    plt.hist(main_descendants, bins=range(min(main_descendants), max(main_descendants)))
    plt.savefig(main_arguments.image) if main_arguments.image is not None else plt.show()
