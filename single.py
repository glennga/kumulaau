#!/usr/bin/env python3
from numpy.random import poisson, choice
from numpy import ndarray, array
from numba import jit


class Population:
    """ Defines the population class, which evolves a single population of 2N alleles given a common ancestor and
    mutation rates. """

    def __init__(self, mu_u: float, mu_d: float, i_0: int, kappa: int = 0):
        """ Constructor. A upward mutation rate mu_u and downward mutation rate mu_d must be specified, each of which
        must exist in [0, 1]. Ancestor length must also be specified. A stationary lower bound kappa can be specified,
        which prevents any additional mutations from occurring if the length moves toward this.

        :param mu_u: Upward mutation rate. Must exist in [0, 1].
        :param mu_d: Downward mutation rate. Must exist in [0, 1].
        :param i_0: Start of the population chain. Specifies the original ancestor.
        :param kappa: Stationary lower bound on repeat lengths.
        """
        assert 0 <= mu_u <= 1 and 0 <= mu_d <= 1
        self.mu_u, self.mu_d, self.kappa, self.i_0 = mu_u, mu_d, kappa, i_0

        # The current ancestral chain as an array, with older generations existing in earlier indices.
        self.ell = None

        # We start with zero mutations, and our current generation holds only the ancestor.
        self.n_mu, self.ell_last = 0, array([self.i_0])

    @staticmethod
    @jit(nopython=True, nogil=True)
    def mutate(ell: ndarray, g: int, n: int, mu_u: float, mu_d: float, kappa: int) -> None:
        """ Mutate some set of individual with the given mutation parameters by drawing the total number of mutations
        from a Poisson process. The number of generations is determined by 2N / g'th triangle number.

        :param ell: Evolution chain to draw ancestors from and save descendants to.
        :param g: Current distance of chain from start. This serves as input into the triangle number generator.
        :param n: Effective population size. There exists 2N alleles.
        :param mu_u: Upward mutation rate. Must exist in [0, 1].
        :param mu_d: Downward mutation rate. Must exist in [0, 1].
        :param kappa: Stationary lower bound on repeat lengths.
        """
        # Define our triangle number generator here.
        triangle = lambda a: int(a * (a + 1) / 2)

        # Start and end of the history chains for the descendants.
        start_desc, end_desc = triangle(g + 1), (triangle(g + 1) + g + 2)

        # Draw the direct ancestors, and create an array of descendants from ancestors.
        direct_ancestors = ell[triangle(g):start_desc]
        ell[start_desc:end_desc] = array([choice(direct_ancestors) for _ in range(g + 2)])

        # TODO: Find a workaround that doesn't incur a race condition to count number of mutations.
        # # Add this to our mutation count.
        # self.n_mu = self.n_mu + x_u + x_d

        # If we reach our lower bound, do not mutate further.
        clamp = lambda a: a if a > kappa else kappa

        for j in range(len(ell[start_desc:end_desc])):
            # Compute the number of upward and downward mutations that have occurred during this period.
            x_u = poisson(mu_u * (2 * n / triangle(g + 1)) * ell[start_desc + j] / 2.0)  # Divide two distributions.
            x_d = poisson(mu_d * (2 * n / triangle(g + 1)) * ell[start_desc + j] / 2.0)

            # Append our mutation changes to the evolution history chain.
            ell[start_desc + j] = clamp(x_u - x_d + ell[start_desc + j])

    def evolve(self, n: int) -> ndarray:
        """ Evolve our ancestral populace to n (effective population size) individual satellite nodes.

        :param n: Effective population size. There exists 2N alleles.
        :return: Numpy array of repeat lengths evolved to a stationary distribution from the ancestral population.
        """
        # Define our triangle number generator here.
        triangle = lambda a: int(a * (a + 1) / 2)

        # Reserve triangle(2*n) elements for our ancestor chain.
        self.ell = array([0 for _ in range(triangle(2*n))])
        self.ell[0] = self.i_0

        # Iterate through 2N - 1 generations, which represent periods of coalescence. Perform our mutation process.
        [self.mutate(self.ell, g, n, self.mu_u, self.mu_d, self.kappa) for g in range(2*n - 1)]

        # Return the current generation of ancestors.
        self.ell_last = self.ell[-2*n:]
        return self.ell_last


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pprint import pprint

    parser = ArgumentParser(description='Simulate the evolution of single population top-down (ancestor first).')
    parser.add_argument('-kappa', help='Repeat length stationary lower bound.', type=int, default=0)
    parser.add_argument('-n', help='Effective population size.', type=int)
    parser.add_argument('-i_0', help='Repeat length of starting ancestor.', type=int)
    parser.add_argument('-mu_u', help='Upward mutation rate. Bounded in [0, 1].', type=float)
    parser.add_argument('-mu_d', help='Downward mutation rate. Bounded in [0, 1].', type=float)
    args = parser.parse_args()  # Parse our arguments.

    # Print the results of evolving our ancestors.
    pprint([x for x in Population(mu_u=args.mu_u, mu_d=args.mu_d, i_0=args.i_0, kappa=args.kappa).evolve(args.n)])
