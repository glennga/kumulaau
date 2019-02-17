#!/usr/bin/env python3
from __future__ import annotations

from numpy import ndarray, array, arange, empty, nextafter, asarray
from numpy.random import uniform, choice, shuffle, exponential, poisson
from collections import Callable
from argparse import Namespace
from numba import jit, prange
import pop


class BaseParameters(object):
    def __init__(self, n: float, f: float, c: float, d: float, kappa: float, omega: float):
        """ Constructor. This is just meant to be a data class for the mutation model. Note that if any changes are made
        to the quantity or type of parameters here, they MUST be changed in "_pop.c" as well.

        :param n: Population size, used for determining the number of generations between events.
        :param f: Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        """
        self.n, self.f, self.c, self.d, self.kappa, self.omega = \
            round(n), f, c, d, round(kappa), round(omega)

        self.PARAMETER_COUNT = 6

    def __iter__(self):
        """ Return each our of parameters in the following order: n, f, c, d, kappa, omega

        :return: Iterator for all of our parameters.
        """
        for parameter in [self.n, self.f, self.c, self.d, self.kappa, self.omega]:
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
                                  d=arguments.d,
                                  kappa=arguments.kappa,
                                  omega=arguments.omega)
        else:
            return BaseParameters(n=arguments.n_sigma,
                                  f=arguments.f_sigma,
                                  c=arguments.c_sigma,
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
                                            d=walk(theta.d, pi_sigma.d),
                                            kappa=walk(theta.kappa, pi_sigma.kappa),
                                            omega=walk(theta.omega, pi_sigma.omega))

            if theta_proposed.n > 0 and \
                    theta_proposed.f >= 0 and \
                    theta_proposed.c > 0 and \
                    theta_proposed.d >= 0 and \
                    0 < theta_proposed.kappa < theta_proposed.omega:
                break

        return theta_proposed


class Population(object):
    def __init__(self, theta: BaseParameters, accel_c: bool=True):
        """ Constructor. Perform the bound checking for each parameter here.

        :param theta: Parameter models to use to evolve our population.
        :param accel_c: Use the C extension '_pop.c' to trace our tree and evolve. Defaults to true.
        """
        # Ensure that all variables are bounded properly.
        theta.n = max(theta.n, 0)
        theta.f = max(theta.f, 0)
        theta.c = max(theta.c, nextafter(0, 1))  # Dr. Reed gave a strict bound > 0 here, but I forget why...
        theta.d = max(theta.d, 0)
        theta.kappa = max(theta.kappa, 0)
        theta.omega = max(theta.omega, theta.kappa)
        self.theta = theta

        self.accel_c = accel_c  # Remember our decision to use the C extension or not.
        self.offset = 0
        self.is_evolved = False
        self.evolved = empty(2 * self.theta.n)

        if not accel_c:  # Trace our tree. We do not perform repeat length determination at this step.
            self.coalescent_tree = empty([self.triangle(2 * self.theta.n)], dtype='int')
            self._trace_tree(self.coalescent_tree, self.theta.n, self._coalesce, self.triangle)
        else:
            self.population_tree_pointer = pop.trace(theta.n, theta.f, theta.c, theta.d, theta.kappa, theta.omega)

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
    def _mutate_generation(t: int, ell: int, c: float, d: float, kappa: int, omega: int) -> int:
        """ Given a repeat length 'ell', we mutate this repeat length up or down dependent on our parameters
        c (upward constant bias) and d (downward linear bias). If we reach our lower bound kappa, we do not mutate
        further. Optimized by Numba. The mutation model gives us the following focal bias:

        \hat{L} = \frac{-c}{-d}.

        We repeat this for t generations.

        :param t: The number of generations to mutate for.
        :param ell: The current repeat length to mutate.
        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of our repeat length space. If 'ell = kappa', we do not mutate.
        :param omega: Upper bound of our repeat length space.
        :return: A mutated repeat length.
        """
        for _ in range(t):
            # If we reached some value kappa, we do not mutate any further.
            if ell == kappa:
                return ell

            # Compute our upward mutation rate. We are bounded by omega.
            ell = min(omega, ell + 1) if uniform(0, 1) < c else ell

            # Compute our downward mutation rate. We are bounded by kappa.
            ell = max(kappa, ell - 1) if uniform(0, 1) < ell * d else ell

        # Return the result of mutating over several generations.
        return ell

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu')
    def _mutate_draw(t: int, ell: int, c: float, d: float, kappa: int, omega: int) -> int:
        """ Given a repeat length 'ell', we mutate this repeat length up or down dependent on our parameters
        c (upward constant bias) and d (downward linear bias). If we reach our lower bound kappa, we do not mutate
        further. Optimized by Numba. The mutation model gives us the following focal bias:

        \hat{L} = \frac{-c}{-d}.

        As opposed to the "_mutate_generation" function, we draw the number of upward mutations and downward mutations,
        then return the sum. We perform this in constant time as opposed to $\Theta (t)$.

        :param t: The number of generations to mutate for.
        :param ell: The current repeat length to mutate.
        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of our repeat length space. If 'ell = kappa', we do not mutate.
        :param omega: Upper bound of our repeat length space.
        :return: A mutated repeat length.
        """
        # If we reached some value kappa, we do not mutate any further.
        if ell == kappa:
            return ell

        # Otherwise, compute the difference between our upward draw and downward draw.
        return min(omega, max(kappa, ell + int(poisson(t * c) - poisson(t * ell * d))))

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _coalesce(tau: int, coalescent_tree: ndarray, triangle_f: Callable) -> None:
        """ Generate the current coalescence event by sampling from our ancestors to our descendants. We save the
        indices of our ancestors to our descendants here (like a pointer), as opposed to their length. Optimized by
        Numba.

        :param tau: The current coalescence event to generate (how far along our ancestor chain to generate).
        :param coalescent_tree: Ancestor history chain, whose generations are indexed by triangle numbers.
        :return: None.
        """
        # We save the indices of our ancestors to our descendants.
        coalescent_tree[triangle_f(tau + 1) + 1:triangle_f(tau + 2)] = arange(triangle_f(tau), triangle_f(tau + 1))
        coalescent_tree[triangle_f(tau + 1)] = choice(coalescent_tree[triangle_f(tau + 1) + 1:triangle_f(tau + 2)])
        shuffle(coalescent_tree[triangle_f(tau + 1):triangle_f(tau + 2)])

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _evolve(coalescent_tree: ndarray, tau: int, theta: ndarray, triangle_f: Callable, mutate_f: Callable) -> None:
        """ Given a coalescent tree whose values represent the indexes of their ancestor, perform repeat length
        determination for the tau'th event and generate a repeat length population. This must be perform in sequential
        order. Optimized by Numba.

        :param coalescent_tree: A 1D array containing the tree. We output the lengths to the next generation here.
        :param tau: The numbered coalescent event to perform repeat length determination for.
        :param theta: Our parameters, in the order of: n, f, c, d, kappa, omega.
        :param triangle_f: The triangle (binomial of two) function to use.
        :param mutate_f: The mutation function to use.
        :return: None.
        """
        n, f, c, d, kappa, omega = theta  # Unpack our parameters.

        # We define our descendants for the tau'th coalescent.
        descendants = coalescent_tree[triangle_f(tau + 1):triangle_f(tau + 2)]

        # Determine the time to coalescence. This is exponentially distributed, but the mean stays the same. Scale by f.
        t_coalescence = max(1, round(exponential(f * 2 * n / triangle_f(tau + 1))))

        # Iterate through each of the descendants (currently pointers) and determine each ancestor.
        for k in prange(descendants.size):
            descendant_to_evolve = coalescent_tree[descendants[k]]

            # Evolve each ancestor according to the average time to coalescence and the scaling factor f.
            descendant_to_evolve = mutate_f(t_coalescence, descendant_to_evolve, c, d, kappa, omega)

            # Save our descendant state.
            descendants[k] = descendant_to_evolve

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=False)
    def _trace_tree(coalescent_tree: ndarray, n: int, coalesce_f: Callable, triangle_f: Callable) -> None:
        """ Create a random evolutionary tree. The result is a 1D array, indexed by triangle(n). Repeat length
        determination does not occur at this stage, rather we determine which ancestor belongs to who. Optimized by
        Numba.

        :param coalescent_tree: A 1D array containing the tree to save to.
        :param n: The number of individuals to generate (our sample size, or the number of leaves we have).
        :param coalesce_f: The coalescence function to use to construct our tree.
        :param triangle_f: The triangle (binomial of two) function to use.
        :return: None.
        """
        for tau in range(0, 2 * n - 1):  # Generate 2N coalescence events.
            coalesce_f(tau, coalescent_tree, triangle_f)

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=False)
    def _evolve_n(coalescent_tree: ndarray, theta: ndarray, offset: int, evolve_f: Callable,
                  triangle_f: Callable, mutate_f: Callable) -> None:
        """ Perform repeat length determination for the entire coalescent tree. This overwrites the tree given to us,
        meaning that we can only do this once. The common ancestors for this array should be manually set before
        using this function. Optimized by Numba.

        :param coalescent_tree: A 1D array containing the tree we save our repeat lengths to.
        :param theta: Our parameters, in the order of: n, f, c, d, kappa, omega.
        :param offset: The number of common ancestors located in our tree. Determines where to start.
        :param evolve_f: Function to determine the repeat lengths of the tau'th coalescent event.
        :param triangle_f: The triangle (binomial of two) function to use.
        :param mutate_f: The mutation function to use.
        :return: None.
        """
        n, f, c, d, kappa, omega = theta  # Unpack our parameters.

        for tau in range(offset, int(2 * n - 1)):
            evolve_f(coalescent_tree, tau, theta, triangle_f, mutate_f)

    def evolve(self, i_0: ndarray) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until '2n' alleles are present.

        :param i_0 Array of starting common ancestors.
        :return: The evolved generation from the common ancestor.
        """
        if self.is_evolved:  # If we have evolved, then return what we currently have. Otherwise, raise our flag.
            return self.evolved
        else:
            self.is_evolved = True

        if not self.accel_c:
            # Determine our offset, and seed our ancestors for the tree.
            self.offset = i_0.size - 1
            self.coalescent_tree[self.triangle(self.offset):self.triangle(self.offset + 1)] = i_0

            # From our common ancestors, descend forward in time and populate our tree with repeat lengths.
            k = (self._evolve, self.triangle, self._mutate_draw)
            self._evolve_n(self.coalescent_tree, array(list(self.theta)), self.offset, *k)

            self.evolved = self.coalescent_tree[-2 * self.theta.n:]
        else:
            # Evolve our population using the tree generated at instantiation.
            self.evolved = asarray(pop.evolve(self.population_tree_pointer, [i for i in i_0]))

        # Return the evolved generation of ancestors.
        return self.evolved


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentParser
    from timeit import default_timer as timer
    from numpy import std, average
    from matplotlib import use

    parser = ArgumentParser(description='Simulate the evolution of single population.')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)
    paa('-image', 'Image file to save resulting repeat length distribution (histogram) to.', str)
    parser.add_argument('-accel_c', help='Use C extension to run the simulation. Toggle w/ 1/0.',
                        type=int, choices=[0, 1], default=1)

    parser.add_argument('-i_0', help='Repeat lengths of starting ancestors.', type=int, nargs='+')
    paa('-n', 'Starting population size.', int)
    paa('-f', 'Scaling factor for total mutation rate.', float)
    paa('-c', 'Constant bias for the upward mutation rate.', float)
    paa('-d', 'Linear bias for the downward mutation rate.', float)
    paa('-kappa', 'Lower bound of repeat lengths.', int)
    paa('-omega', 'Upper bound of repeat lengths.', int)
    main_arguments = parser.parse_args()  # Parse our arguments.

    use('TkAgg')  # Use a different backend (plotting to display).
    from matplotlib import pyplot as plt

    # Run once, to separate compilation time from actual running time.
    Population(BaseParameters.from_args(main_arguments)).evolve(array(main_arguments.i_0))

    # Evolve some population 1000 times.
    start_t = timer()
    main_descendants = \
        [Population(BaseParameters.from_args(main_arguments),  main_arguments.accel_c == 1
                    ).evolve(array(main_arguments.i_0)) for _ in range(10000)]
    end_t = timer()
    print('Time Elapsed (10000x): [\n\t' + str(end_t - start_t) + '\n]')

    # Determine the variance and mean of the generated populations.
    print('Mean (10000x): [\n\t' + str(average(main_descendants)) + '\n]')
    print('Deviation (10000x): [\n\t' + str(std(main_descendants)) + '\n]')

    # Display a histogram of the first generated populace.
    plt.hist(main_descendants[0], bins=range(min(main_descendants[0]), max(main_descendants[0])))
    plt.savefig(main_arguments.image) if main_arguments.image is not None else plt.show()
