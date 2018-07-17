#!/usr/bin/env python3
from numpy.random import uniform, choice
from numpy import ndarray, array, log, power, floor
from numba import jit, prange


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
    return kappa + floor(log(1 - uniform(0, 1) * (1 - power(1 - m, omega - kappa))) / log(1 - m))


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


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def coalesce_n(c: int, ell: ndarray, big_n: int, mu: float, s: float, kappa: int, omega: int, u: float, v: float,
               m: float, p: float) -> None:
    """ Simulate the mutation of 'c' coalescence events, and store the results in our history chain. Optimized by
    Numba.

    :param c: Number of coalescence events + 1 to generate. Represents the current distance of the chain from the start.
    :param ell: Ancestor history chain, whose generations are indexed by triangle numbers.
    :param big_n: Effective population size, used for determining the number of generations between events.
    :param mu: Mutation rate, bounded by (0, infinity).
    :param s: Proportional rate (to repeat length), bounded by (-1 / (omega - kappa + 1), infinity).
    :param kappa: Lower bound of possible repeat lengths (minimum of state space).
    :param omega: Upper bound of possible repeat lengths (maximum of state space).
    :param u: Constant bias parameter, used to determine the probability of an expansion and bounded by [0, 1].
    :param v: Linear bias parameter, used to determine the probability of an expansion.
    :param m: Success probability for the truncated geometric distribution, bounded by [0, 1].
    :param p: Probability that the geometric distribution will be used vs. single repeat length mutations.
    :return: None.
    """
    # Determine the range of our ancestors and our output range (indices of descendants).
    start_anc, end_anc = triangle_n(c), triangle_n(c + 1)
    start_desc, end_desc = triangle_n(c + 1), triangle_n(c + 2)

    # Determine the repeat lengths for the new generation before mutation is applied (draw with replacement).
    ell[start_desc:end_desc] = array([choice(ell[start_anc:end_anc]) for _ in range(c + 2)])

    # Iterate through each of the descendants and apply the mutation.
    for a in prange(end_desc - start_desc):
        for _ in range(max(1, round(2 * big_n / triangle_n(c + 1)))):
            ell[start_desc + a] = mutate_n(ell[start_desc + a], mu, s, kappa, omega, u, v, m, p)


class ModelParameters(object):
    def __init__(self, i_0: int, big_n: int, mu: float, s: float, kappa: int, omega: int, u: float, v: float,
                 m: float, p: float):
        """ Constructor. This is just meant to be a data class for the mutation model.

        :param i_0: Repeat length of the common ancestor.
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


class Single:
    def __init__(self, m_p: ModelParameters):
        """ Constructor. Perform the bound checking for each parameter here.

        :param m_p: Parameter models to use to evolve our population.
        """
        from numpy import empty

        assert m_p.i_0 > 0 and m_p.big_n > 0 and (-1 / (m_p.omega - m_p.kappa + 1)) < m_p.s and 0 < m_p.mu
        assert 0 <= m_p.u <= 1 and 0 <= m_p.m <= 1 and 0 <= m_p.p <= 1

        self.i_0, self.big_n, self.omega, self.kappa, self.s, self.mu, self.u, self.v, self.m, self.p \
            = m_p.i_0, m_p.big_n, m_p.omega, m_p.kappa, m_p.s, m_p.mu, m_p.u, m_p.v, m_p.m, m_p.p

        # Define our ancestor chain, and the array that will hold the end population after 'evolve' is called.
        self.ell = empty([self._triangle(2 * m_p.big_n)], dtype='int')
        self.ell[0], self.ell_evolved = m_p.i_0, empty([m_p.big_n], dtype='int')

    @staticmethod
    def _triangle(a):
        """ Triangle number generator. Given 'a', return a choose 2. Using the Numba optimized version.

        :param a: Which triangle number to return.
        :return: The a'th triangle number.
        """
        return triangle_n(a)

    def _coalesce(self, c: int) -> None:
        """ Simulate the mutation of 'c' coalescence events, and store the results in our history chain. Using the
        optimized Numba version.

        :param c: Number of coalescence events to generate. Represents the current distance of the chain from the start.
        :return: None.
        """
        coalesce_n(c, self.ell, self.big_n, self.mu, self.s, self.kappa, self.omega, self.u, self.v, self.m, self.p)

    def evolve(self) -> ndarray:
        """ Using the common ancestor 'i_0', evolve the population until 'big_n' individuals are present.

        :return: The evolved generation from the common ancestor.
        """
        # Iterate through 2N - 1 generations, which represent periods of coalescence. Perform our mutation process.
        [self._coalesce(c) for c in range(2 * self.big_n - 1)]

        # Return the current generation of ancestors.
        self.ell_evolved = self.ell[-2 * self.big_n:]
        return self.ell_evolved


if __name__ == '__main__':
    from argparse import ArgumentParser
    from matplotlib import pyplot as plt

    parser = ArgumentParser(description='Simulate the evolution of single population top-down (ancestor first).')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)
    
    paa('-i_0', 'Repeat length of starting ancestor.', int)
    paa('-big_n', 'Effective population size.', int)
    paa('-mu', 'Mutation rate, bounded by (0, infinity).', float)
    paa('-s', 'Proportional rate, bounded by (-1 / (omega - kappa + 1), infinity).', float)
    paa('-kappa', 'Lower bound of possible repeat lengths.', int)
    paa('-omega', 'Upper bound of possible repeat lengths.', int)
    paa('-u', 'Constant bias parameter, bounded by [0, 1].', float)
    paa('-v', 'Linear bias parameter, bounded by (-infinity, infinity).', float)
    paa('-m', 'Success probability for truncated geometric distribution.', float)
    paa('-p', 'Probability that the repeat length change is +/- 1.', float)
    args = parser.parse_args()  # Parse our arguments.

    # Display the results of evolving our ancestors.
    pop = Single(ModelParameters(i_0=args.i_0, big_n=args.big_n, mu=args.mu, s=args.s, kappa=args.kappa,
                                 omega=args.omega, u=args.u, v=args.v, m=args.m, p=args.p))
    pop.evolve()

    # Display a histogram.
    plt.hist(pop.ell_evolved, bins=range(args.kappa, args.omega)), plt.show()
