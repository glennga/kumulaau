#!/usr/bin/env python3
from typing import Any, List

# TODO: Optimization for the future, use numpy arrays and Numba with the GPU.


# class S:
#     """ Defines a single satellite node. There exists a field for the current generation, the associated repeat
#     length, and a pointer to it's ancestor (if we want to walk trees. """
#
#     def __init__(self, g: int, rl: int, anc):  # TODO: Type hints don't work for the class itself?
#         """ Constructor. Define the current generation associated with this node TODO: Finish this.
#
#         :param g:
#         :param rl:
#         :param anc:
#         """
#         self.g, self.rl, self.anc = g, rl, anc


class P:
    """ Defines the population class, which evolves ... TODO: Finish this. """

    def __init__(self, mu_u: float, mu_d: float, i_0: List[int], kappa: int = 0):
        """ Constructor. A upward mutation rate mu_u and downward mutation rate mu_d must be specified, each of which
        must exist in [0, 1]. Ancestor(s) must also be specified. A stationary lower bound kappa can be specified,
        which prevents any additional mutations from occurring if the length moves toward this.

        :param mu_u: Upward mutation rate. Must exist in [0, 1].
        :param mu_d: Downward mutation rate. Must exist in [0, 1].
        :param i_0: Start of the population chain. Specifies the original ancestors.
        :param kappa: Stationary lower bound on repeat lengths.
        """
        assert 0 <= mu_u <= 1 and 0 <= mu_d <= 1
        self.mu_u, self.mu_d, self.kappa, self.i_0 = mu_u, mu_d, kappa, i_0
        self.ell = None  # The current ancestral chain as an array, with older generations existing in earlier indices.

    def mutate(self, i_t: int, h: int, n: int) -> int:
        """ Mutate some individual with the given mutation parameters by drawing the total number of mutations from a
        Poisson process. The number of generations is determined by 2N / h'th triangle number.

        :param i_t: Parent length to evolve from.
        :param h: Current tree depth. Determines the generation of the resulting node.
        :param n: Effective population size, used to determine the number of resulting mutations.
        :return: The mutated repeat length, given 2N / h'th triangle number generations to mutate from.
        """
        from numpy.random import poisson
        clamp = lambda x: 0 if x < 0 else x  # Range: [0, infinity)

        delta_i = h * (h + 1) / 2  # Generate the i'th triangle number.

        # Compute the number of upward and downward mutations. \lambda = g * \mu.
        x_d, x_u = poisson((n / (1 + int(delta_i))) * self.mu_u), poisson((n / (1 + int(delta_i))) * self.mu_d)

        # Return a new list, whose repeat length is the sum of the mutation counts above.
        return clamp(i_t + x_u - x_d)

    def evolve(self, n: int) -> Any:
        """ Evolve our ancestral populace to n (effective population size) individual satellite nodes.

        :param n: Effective population size. The length of the resulting populace to be returned.
        :return: Numpy array of repeat lengths evolved to a stationary distribution from the ancestral population.
        """
        from numpy.random import choice
        from numpy import array

        # Uhh... Not too efficient, but it works lol. TODO: Figure out the number of nodes in the resulting tree.
        m = 0
        for j in range(1, n):
            for k in range(j + 1):
                m += 1

        self.ell = array([0 for _ in range(m)])  # Define our ancestor chain.
        self.ell[0:len(self.i_0)] = array(self.i_0)

        # TODO: Draw this shit out, it's not making sense right now.
        for j in range(1, n):
            ell_t = self.ell[(j - 1)*(j - 1):]  # From our ancestor chain, choose the last j individuals.
            for k in range(j + 1):  # We want j + 1 = k individuals. Mutate each of these.
                self.ell[j*(j - 1):(j + 1)*(j - 1)] = self.mutate(choice(ell_t), k, n)

        return self.anc[-n:]  # Return the last n individuals in our chain.


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pprint import pprint
    from numpy import array

    parser = ArgumentParser(description='Simulate the evolution of single population top-down (ancestor first).')
    parser.add_argument('-mu_u', help='Upward mutation rate. Bounded in [0, 1].', type=float)
    parser.add_argument('-mu_d', help='Downward mutation rate. Bounded in [0, 1].', type=float)
    parser.add_argument('-kappa', help='Repeat length stationary lower bound.', type=int, default=0)
    parser.add_argument('-n', help='Effective population size.', type=int)
    parser.add_argument('-i_0', help='List of repeat lengths of starting ancestors.', type=int, nargs='+')
    args = parser.parse_args()  # Parse our arguments.

    # Print the results of evolving our ancestor eff_n times.
    pprint([x.rl for x in P(mu_u=args.mu_u, mu_d=args.mu_d, i_0=args.i_0, kappa=args.kappa).evolve(args.n)])
