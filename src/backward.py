#!/usr/bin/env python3
from mutate import Mutate, triangle_n, mutate_n
from numpy.random import choice, shuffle
from numpy import ndarray, arange
from numba import jit, prange


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def sample_nr(ancestors: ndarray, descendants: ndarray) -> None:
    """ Randomly sample |descendents| from ancestors vector without replacement. Only one ancestor is able have
    two descendants.

    :param ancestors: Vector of the ancestors to choose from.
    :param descendants: Output array to store our ancestors to.
    :return: None.
    """
    # Choose the indices of our ancestors to save to.
    descendants[1:] = (arange(ancestors.size))
    descendants[0] = choice(descendants[1:])
    shuffle(descendants)

    # Use the indexes to choose which descendant holds which ancestor.
    for i in prange(descendants.size):
        descendants[i] = ancestors[descendants[i]]


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

    # Determine the repeat lengths for the new generation before mutation is applied (draw without replacement).
    sample_nr(ell[start_anc:end_anc], ell[start_desc:end_desc])

    # Iterate through each of the descendants and apply the mutation.
    for a in prange(end_desc - start_desc):
        for _ in range(max(1, round(2 * big_n / triangle_n(c + 1)))):
            ell[start_desc + a] = mutate_n(ell[start_desc + a], mu, s, kappa, omega, u, v, m, p)


class Backward(Mutate):
    def _coalesce(self, c: int) -> None:
        """ Simulate the mutation of 'c' coalescence events, and store the results in our history chain. Using the
        optimized Numba version.

        :param c: Number of coalescence events to generate. Represents the current distance of the chain from the start.
        :return: None.
        """
        coalesce_n(c, self.ell, self.big_n, self.mu, self.s, self.kappa, self.omega, self.u,
                   self.v, self.m, self.p)

    def evolve(self) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until 'big_n' individuals are present.

        :return: The evolved generation from the common ancestor.
        """
        # First, create the evolutionary tree.

        # Iterate through 2N - 1 generations, which represent periods of coalescence. Perform our mutation process.
        [self._coalesce(c) for c in range(self.offset, 2 * self.big_n - 1)]

        # Return the current generation of ancestors.
        self.ell_evolved = self.ell[-2 * self.big_n:]
        return self.ell_evolved
