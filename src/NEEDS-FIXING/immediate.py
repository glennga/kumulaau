#!/usr/bin/env python3
from mutate import BaseParameters, Mutate, triangle_n, mutate_n
from numpy.random import choice, shuffle
from numpy import ndarray, arange
from numba import jit, prange


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def sample_n(ancestors: ndarray, descendants: ndarray) -> None:
    """ Randomly sample |descendants| from ancestors vector without replacement. Only one ancestor is able have
    two descendants.

    :param ancestors: Vector of the ancestors to choose from.
    :param descendants: Output array to store our ancestors to.
    :return: None.
    """
    # Choose the indices of our ancestors to save to.
    descendants[1:] = arange(ancestors.size)
    descendants[0] = choice(descendants[1:])
    shuffle(descendants)

    # Use the indexes to choose which descendant holds which ancestor.
    for i in prange(descendants.size):
        descendants[i] = ancestors[descendants[i]]


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def coalesce_n(c: int, ell: ndarray, big_n: int, mu: float, f: float, s: float, kappa: int, omega: int, u: float,
               v: float, m: float, p: float) -> None:
    """ Simulate the mutation of 'c' coalescence events, and store the results in our history chain. Optimized by
    Numba.

    :param c: Number of coalescence events + 1 to generate. Represents the current distance of the chain from the start.
    :param ell: Ancestor history chain, whose generations are indexed by triangle numbers.
    :param big_n: Effective population size, used for determining the number of generations between events.
    :param mu: Mutation rate, bounded by (0, infinity).
    :param f: Scaling factor for mutation. Smaller = shorter time to coalescence.
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

    # Determine the repeat lengths for the new generation before mutation is applied (draw without replacement).
    sample_n(ell[start_anc:end_anc], ell[start_desc:end_desc])

    # Iterate through each of the descendants and apply the mutation.
    for a in prange(end_desc - start_desc):
        for _ in range(max(1, round(f * 2 * big_n / triangle_n(c + 1)))):
            ell[start_desc + a] = mutate_n(ell[start_desc + a], mu, f, s, kappa, omega, u, v, m, p)


class Immediate(Mutate):
    def __init__(self, parameters: BaseParameters):
        """ Constructor. In addition to the bounds checking, we seed our ancestors with our ancestors.

        :param parameters: Parameter models to use to evolve our population.
        """
        super(Immediate, self).__init__(parameters)

        # Our chain offset is determined by the number of ancestors we have.
        self.offset = len(parameters.i_0) - 1

        # Seed our ancestors. Place them in the proper place of the coalescence tree (dependent on i_0 size).
        self.ell[self._triangle(self.offset):self._triangle(self.offset + 1)] = parameters.i_0

    def _coalesce(self, c: int) -> None:
        """ Simulate the mutation of coalescence event 'c', and store the results in our history chain. Using the
        optimized Numba version.

        :param c: The coalescence event to generate. Represents the current distance of the chain from the start.
        :return: None.
        """
        coalesce_n(c, self.ell, self.big_n, self.mu, self.s, self.kappa, self.omega, self.u,
                   self.v, self.m, self.p)

    def evolve(self, i_0: ndarray=None) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until 'big_n' individuals are present.

        :param i_0 Array of common ancestors. If this parameter is defined, we use this over our constructor 'i_0'.
        :return: The evolved generation from the common ancestor.
        """
        if i_0 is not None:  # Determine our offset and seed our ancestors.
            self.offset = i_0.size - 1
            self.ell[self._triangle(self.offset):self._triangle(self.offset + 1)] = i_0

        # Iterate through 2N - 1 generations, which represent periods of coalescence. Perform our mutation process.
        [self._coalesce(c) for c in range(self.offset, 2 * self.big_n - 1)]

        # Return the current generation of ancestors.
        self.ell_evolved = self.ell[-2 * self.big_n:]
        return self.ell_evolved
