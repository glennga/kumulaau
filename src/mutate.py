#!/usr/bin/env python3
from abc import ABC, abstractmethod
from numpy.random import uniform
from numpy import ndarray, array, log, power, floor
from numba import jit


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def random_alleles(omega: int, kappa: int, n: int) -> ndarray:
    """ Generate n random alleles. This is meant to simulate individuals who do not originate from our ancestor
    set. Optimized by Numba.

    :param omega: Upper bound of possible repeat lengths (maximum of state space).
    :param kappa: Lower bound of possible repeat lengths (minimum of state space).
    :param n: Number of alleles to generate.
    :return: A one dimensional array of random repeat lengths.
    """
    return uniform(omega, kappa, n)


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def triangle_n(a: int) -> int:
    """ Triangle number generator. Given 'a', return a choose 2. Optimized by Numba.

    :param a: Which triangle number to return.
    :return: The a'th triangle number.
    """
    return int(a * (a + 1) / 2)


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def gamma_n(omega: int, kappa: int, m: float) -> int:
    """ Draw from the truncated geometric distribution bounded by omega and kappa, to obtain the number of
    expansions or contractions for mutations greater than 1. Optimized by Numba. Equation found here:
    https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution

    :param omega: Upper bound of possible repeat lengths (maximum of state space).
    :param kappa: Lower bound of possible repeat lengths (minimum of state space).
    :param m: Success probability for the truncated geometric distribution, bounded by [0, 1].
    :return: The number of contractions or expansions.
    """
    return floor(log(1 - uniform(0, 1) * (1 - power(1 - m, omega - kappa))) / log(1 - m))


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def beta_n(i: int, mu: float, kappa: int, s: float) -> float:
    """ Determine the mutation rate of some allele i, which is dependent on the current repeat length. Optimized by
    Numba.

    :param i: Repeat length of the allele to mutate.
    :param mu: Mutation rate, bounded by (0, infinity).
    :param kappa: Lower bound of possible repeat lengths (minimum of state space).
    :param s: Proportional rate (to repeat length), bounded by (-1 / (omega - kappa + 1), infinity).
    :return: Mutation rate given the current repeat length.
    """
    return mu * (1 + (i - kappa) * s)


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def alpha_n(i: int, u: float, v: float, kappa: int) -> float:
    """ Determine the probability that a mutation results in a expansion. The probability that a mutation results
    in a contraction is (1 - alpha). Optimized by Numba.

    The focal bias 'f' is defined as: f = ((u - 0.5) / v) + kappa

    :param i: Repeat length of the allele to mutate.
    :param u: Constant bias parameter, used to determine the probability of an expansion and bounded by [0, 1].
    :param v: Linear bias parameter, used to determine the probability of an expansion.
    :param kappa: Lower bound of possible repeat lengths (minimum of state space).
    :return: Probability that the current allele will expand.
    """
    return max(0.0, min(1.0, u - v * (i - kappa)))


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def mutate_n(i: int, mu: float, s: float, kappa: int, omega: int, u: float, v: float, m: float, prob_p: float) -> int:
    """ Given some allele of length i, contract or expand to this given the parameters of the model. Optimized by Numba.

    :param i: Repeat length of ancestor to mutate with.
    :param mu: Mutation rate, bounded by (0, infinity).
    :param s: Proportional rate (to repeat length), bounded by (-1 / (omega - kappa + 1), infinity).
    :param kappa: Lower bound of possible repeat lengths (minimum of state space).
    :param omega: Upper bound of possible repeat lengths (maximum of state space).
    :param u: Constant bias parameter, used to determine the probability of an expansion and bounded by [0, 1].
    :param v: Linear bias parameter, used to determine the probability of an expansion.
    :param m: Success probability for the truncated geometric distribution, bounded by [0, 1].
    :param prob_p: Probability that the geometric distribution will be used vs. single repeat length mutations.
    :return: 'j', or the mutated repeat length from 'i'.
    """
    # Compute mutation rate and the probability of expansion.
    beta_i, alpha_i = beta_n(i, mu, kappa, s), alpha_n(i, u, v, kappa)

    # Determine if a mutation occurs or not.
    y_1 = (1 if uniform(0, 1) < beta_i else 0)

    # Determine if a contraction or expansion occurs.
    y_2 = (1 if uniform(0, 1) < alpha_i else -1)

    # Determine the length of the contraction or expansion.
    y_3 = (gamma_n(omega, kappa, m) if uniform(0, 1) < prob_p else 1)

    # Determine the new length. Restrict lengths to [kappa, omega]. Note that if kappa is reached, an allele stays.
    return i if (i == kappa) else max(kappa, min(omega, (y_1 * y_2 * y_3) + i))


class ModelParameters(object):
    def __init__(self, i_0: ndarray, big_n: int, mu: float, s: float, kappa: int, omega: int, u: float, v: float,
                 m: float, p: float, ):
        """ Constructor. This is just meant to be a data class for the mutation model.

        :param i_0: Repeat lengths of the common ancestors.
        :param big_n: Effective population size, used for determining the number of generations between events.
        :param mu: Mutation rate, bounded by (0, infinity).
        :param s: Proportional rate (to repeat length), bounded by (-1 / (omega - kappa + 1), infinity).
        :param kappa: Lower bound of possible repeat lengths (minimum of state space).
        :param omega: Upper bound of possible repeat lengths (maximum of state space).
        :param u: Constant bias parameter, used to determine the probability of an expansion and bounded by [0, 1].
        :param v: Linear bias parameter, used to determine the probability of an expansion.
        :param m: Success probability for the truncated geometric distribution, bounded by [0, 1].
        :param p: Probability that the geometric distribution will be used vs. single repeat length mutations.
        """
        self.i_0, self.big_n, self.omega, self.kappa, self.s, self.mu, self.u, self.v, self.m, self.p \
            = i_0, big_n, omega, kappa, s, mu, u, v, m, p

        self.PARAMETER_COUNT = 10

    def __iter__(self):
        """ Return each our of parameters in the following order: big_n, mu, s, kappa, omega, u, v, m, p,

        :return: Iterator for all of our parameters.
        """
        for parameter in [self.big_n, self.mu, self.s, self.kappa, self.omega, self.u, self.v, self.m, self.p]:
            yield parameter

    def __len__(self):
        """ The number of parameters that exist here.

        :return: The number of parameters we have.
        """
        return self.PARAMETER_COUNT


class Mutate(ABC):
    @staticmethod
    def _triangle(a):
        """ Triangle number generator. Given 'a', return a choose 2. Using the Numba optimized version.

        :param a: Which triangle number to return.
        :return: The a'th triangle number.
        """
        return triangle_n(a)

    def __init__(self, parameters: ModelParameters):
        """ Constructor. Perform the bound checking for each parameter here.

        :param parameters: Parameter models to use to evolve our population.
        """
        from numpy import empty

        # Ensure that all variables are bounded properly.
        parameters.i_0 = array([max(min(x, parameters.omega), parameters.kappa) for x in parameters.i_0])
        parameters.big_n = max(parameters.big_n, 0)
        parameters.s = max(parameters.s, (-1 / (parameters.omega - parameters.kappa + 1)))
        parameters.mu = max(parameters.mu, 0.0)
        parameters.u = max(min(parameters.u, 1.0), 0.0)
        parameters.m = max(min(parameters.m, 1.0), 0.0)
        parameters.p = max(min(parameters.p, 1.0), 0.0)

        self.i_0, self.big_n, self.omega, self.kappa, self.s, self.mu, self.u, self.v, self.m, self.p \
            = parameters.i_0, parameters.big_n, parameters.omega, parameters.kappa, parameters.s, \
            parameters.mu, parameters.u, parameters.v, parameters.m, parameters.p

        # Our chain offset is determined by the number of ancestors we have.
        self.offset = len(parameters.i_0) - 1

        # Define our ancestor chain, and the array that will hold the end population after 'evolve' is called.
        self.ell = empty([self._triangle(2 * parameters.big_n) + self.offset], dtype='int')
        self.ell_evolved = empty(([parameters.big_n]), dtype=int)

        # Seed our ancestors. Place them in the proper place of the coalescence tree (dependent on i_0 size).
        self.ell[self._triangle(self.offset):self._triangle(self.offset + 1)] = parameters.i_0

    @abstractmethod
    def evolve(self) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until 'big_n' individuals are present.

        :return: The evolved generation from the common ancestor.
        """
        raise NotImplementedError


if __name__ == '__main__':
    from argparse import ArgumentParser
    from matplotlib import pyplot as plt
    from forward import Forward
    from backward import Backward

    parser = ArgumentParser(description='Simulate the evolution of single population.')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)
    paa('-image', 'Image file to save resulting repeat length distribution (histogram) to.', str)
    parser.add_argument('-sim', help='Type of simulation- (forward or backward).', type=str,
                        choices=['FORWARD', 'BACKWARD'])

    parser.add_argument('-i_0', help='Repeat lengths of starting ancestors.', type=int, nargs='+')
    paa('-big_n', 'Effective population size.', int)
    paa('-mu', 'Mutation rate, bounded by (0, infinity).', float)
    paa('-s', 'Proportional rate, bounded by (-1 / (omega - kappa + 1), infinity).', float)
    paa('-kappa', 'Lower bound of possible repeat lengths.', int)
    paa('-omega', 'Upper bound of possible repeat lengths.', int)
    paa('-u', 'Constant bias parameter, bounded by [0, 1].', float)
    paa('-v', 'Linear bias parameter, bounded by (-infinity, infinity).', float)
    paa('-m', 'Success probability for truncated geometric distribution.', float)
    paa('-p', 'Probability that the repeat length change is +/- 1.', float)
    args, pop = None, parser.parse_args()  # Parse our arguments.

    # Generate our parameters.
    theta = ModelParameters(i_0=array(args.i_0), big_n=args.big_n, mu=args.mu, s=args.s, kappa=args.kappa,
                            omega=args.omega, u=args.u, v=args.v, m=args.m, p=args.p)

    # Evolve, forward or backwards.
    if args.sim.casefold() == 'forward':
        pop = Forward(theta)
    elif args.sim.casefold() == 'backward':
        pop = Backward(theta)
    pop.evolve()

    # Display a histogram.
    plt.hist(pop.ell_evolved, bins=range(args.kappa, args.omega))
    plt.savefig(args.image) if args.image is not None else plt.show()