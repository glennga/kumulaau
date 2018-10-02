#!/usr/bin/env python3
from mutate import BaseParameters, Mutate, triangle_n, mutate_n
from numpy.random import choice, shuffle
from numpy import ndarray, arange, array
from numba import jit, prange


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def coalesce_n(c: int, ell: ndarray) -> None:
    """ Generate the current coalescence event by sampling from our ancestors to our descendants. We save the indices
    of our ancestors to our descendants here (like a pointer), as opposed to their length.

    :param c: Number of coalescence events + 1 to generate. Represents the current distance of the chain from the start.
    :param ell: Ancestor history chain, whose generations are indexed by triangle numbers.
    :return: None.
    """
    # Determine the range of our ancestors and our output range (indices of descendants).
    start_anc, end_anc = triangle_n(c), triangle_n(c + 1)
    start_desc, end_desc = triangle_n(c + 1), triangle_n(c + 2)

    # We save the indices of our ancestors to our descendants.
    ell[start_desc + 1:end_desc] = arange(start_anc, end_anc)
    ell[start_desc] = choice(ell[start_desc + 1:end_desc])
    shuffle(ell[start_desc:end_desc])


class Delayed(Mutate):
    def __init__(self, parameters: BaseParameters):
        """ Constructor. In addition to the bounds checking, we trace our tree.

        :param parameters: Parameter models to use to evolve our population.
        """
        super(Delayed, self).__init__(parameters)
        self.i_0 = parameters.i_0  # Save our ancestors for when we evolve. This may change after instantiation.

        # Trace our tree. We do not perform repeat length determination at this step.
        self._trace_tree()

    def _trace_tree(self) -> None:
        """ Trace the evolutionary tree back to one common ancestor. We store the tree as a 1D array, in chronological
        order (ancestors first).

        :return: None.
        """
        # First, we seed our current generation (the one we sample from) with the appropriate indices.
        self.ell[-2 * self.big_n:] = arange(2 * self.big_n)

        # Generate 2N - 1 coalescence events.
        [coalesce_n(c, self.ell) for c in range(0, 2 * self.big_n - 1)]

    def cut_tree(self, tau: int) -> None:
        """ Using the scaling factor f, we determine how long our tree is and perform a cut at tau. We cutoff from most
        recent to least recent, meaning tau represents the distance from the generation we sample.

        :param tau: The distance from the generation we sample.
        :return: None.
        """


        self.ell = self.ell[-int(self.big_n * self.f - tau):]


    def evolve(self, i_0: ndarray=None) -> ndarray:
        """ Using the common ancestors 'i_0', evolve the population until 'big_n' individuals are present.

        :param i_0 Array of starting common ancestors. Our default value is an array = {10}.
        :return: The evolved generation from the common ancestor.
        """
        # Determine our offset value (where we start the chain).
        i_0 = i_0 if i_0 is not None else array([10])  # We default to {10} (why lol, no one knows).
        self.ell[0:i_0.size], self.offset = i_0, i_0.size - 1

        # We iterate forward 2N - 1 generations. This represents periods of coalescence.
        for c in range(self.offset, 2 * self.big_n - 1):
            descendants = self.ell[self._triangle(c + 1):self._triangle(c + 2)]

            # Iterate through our descendants, and perform our mutation process.
            for j in range(descendants.size):
                descendants[j] = mutate_n(self.ell[j], self.mu, self.f, self.s, self.kappa, self.omega, self.u,
                                          self.v, self.m, self.p)

        # Return the current generation of ancestors.
        self.ell_evolved = self.ell[-2 * self.big_n:]
        return self.ell_evolved
