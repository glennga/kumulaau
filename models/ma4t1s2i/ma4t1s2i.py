#!/usr/bin/env python3
from __future__ import annotations

from kumulaau import Parameter, Model, MCMCA
from kumulaau.distance import Cosine
from argparse import Namespace
from numpy import ndarray
from typing import List


class Parameter4T1S2I(Parameter):
    def __init__(self, n_b: int, n_s1: int, n_s2: int, n_e: int, f_b: float, f_s1: float, f_s2: float, f_e: float,
                 alpha: float, c: float, d: float, kappa: int, omega: int):
        """ Constructor. This is just meant to be a data class for the 4T1S2I model.

        :param n_b: Population size of common ancestor population.
        :param n_s1: Population size of common ancestor descendant one.
        :param n_s2: Population size of common ancestor descendant two.
        :param f_b: Scaling factor for common ancestor population.
        :param f_s1: Scaling factor of common ancestor descendant one.
        :param f_s2: Scaling factor of common ancestor descendant two.
        :param f_e: Scaling factor of end population.
        :param alpha: Admixture constant of common ancestor descendant one.
        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        """
        self.n_b, self.n_s1, self.n_s2, self.n_e, self.f_b, self.f_s1, self.f_s2, self.f_e, self.alpha = \
            round(n_b), round(n_s1), round(n_s2), round(n_e), f_b, f_s1, f_s2, f_e, alpha

        super().__init__(c, d, kappa, omega)

    def _validity(self) -> bool:
        """ Determine if a current parameter set is valid. We constrain the following:

        1. A population size is always positive.
        2. Scaling parameters are always greater than / equal to zero.
        3. Mutation model parameters are never negative.
        4. Kappa (our lower bound) is between 0 and our upper bound, omega.
        5. Our admixture parameter is always positive.
        6. n_e < n_s1 + n_s2 < n_e (we cannot have detached common ancestors).

        :return: True if valid. False otherwise.
        """
        return self.n_b > 0 and self.n_s1 > 0 and self.n_s2 > 0 and self.n_e > 0 and \
            self.f_b >= 0 and self.f_s1 >= 0 and self.f_s2 >= 0 and self.f_e >= 0 and \
            self.c > 0 and self.d >= 0 and \
            0 < self.kappa < self.omega and \
            self.alpha > 0 and \
            self.n_e < self.n_s1 + self.n_s2 < self.n_e


class Model4T1S2I(Model):
    def _generate_topology(self, theta: Parameter4T1S2I) -> List:
        """ Generate a 4-element list of pointers to a single tree in the pop module. The order is as follows:

        1. Common ancestor population.
        2. Descendant 1 from common ancestor population.
        3. Descendant 2 from common ancestor population.
        4. End population.

        :param theta: Parameters4T1S2I set to use with tree tracing.
        :return: 4-element list of pointers to pop module C structure.
        """
        return [self.pop_trace(theta.n_b, theta.f_b, theta.c, theta.d, theta.kappa, theta.omega),
                self.pop_trace(theta.n_s1, theta.f_s1, theta.c, theta.d, theta.kappa, theta.omega),
                self.pop_trace(theta.n_s2, theta.f_s2, theta.c, theta.d, theta.kappa, theta.omega),
                self.pop_trace(theta.n_e, theta.f_e, theta.c, theta.d, theta.kappa, theta.omega)]

    def _resolve_lengths(self, i_0: ndarray) -> ndarray:
        """ Generate a list of lengths of our 1T (one total) 0S (zero splits) 0I (zero intermediates) model.

        :return: List of repeat lengths.
        """
        from numpy.random import normal, shuffle
        from numpy import asarray, abs, concatenate

        # Evolve our common ancestor tree.
        anc_out = self.pop_evolve(self.generate_topology_results[0], i_0)

        # Determine our descendant inputs. Normally distributed around alpha.
        split_alpha = int(round(abs(normal(self.theta.alpha, 0.2)) * len(anc_out)))
        shuffle(anc_out)

        # Evolve our descendant populations 1 and 2.
        desc_1_out = self.pop_evolve(self.generate_topology_results[1], anc_out[0:split_alpha])
        desc_2_out = self.pop_evolve(self.generate_topology_results[2], anc_out[:-split_alpha])

        # Evolve our last descendant population.
        return asarray(self.pop_evolve(self.generate_topology_results[3], concatenate((desc_1_out, desc_2_out))))


class MCMC4T1S2I(MCMCA):
    # Set the model name associated with the database.
    MODEL_NAME = "MA4T1S2I"

    # Set the model schema associated with the database.
    MODEL_SCHEME_SQL = "N_B INT, N_S1 INT, N_S2 INT, N_E INT, " \
                       "F_B FLOAT, F_S1 FLOAT, F_S2 FLOAT, F_E FLOAT " \
                       "ALPHA FLOAT, C FLOAT, D FLOAT, KAPPA INT, OMEGA INT"

    # Set the model class to use.
    MODEL_CLASS = Model4T1S2I

    # Set the parameter class to use.
    PARAMETER_CLASS = Parameter4T1S2I

    # Set the distance class to use.
    DISTANCE_CLASS = Cosine

    @staticmethod
    def _walk(theta, pi_epsilon) -> Parameter4T1S2I:
        """ Given some parameter set theta and the sigma parameters pi_epsilon, generate a new parameter set.

        :param theta: Current point to walk from.
        :param pi_epsilon: Distribution parameters (deviation, with mean at theta) to walk with.
        :return: A new parameter set.
        """
        from numpy.random import normal

        # TODO: Figure out how to walk.
        return Parameter4T1S2I(n=round(normal(theta.n, pi_epsilon.n)),
                               f=normal(theta.f, pi_epsilon.f),
                               c=normal(theta.c, pi_epsilon.c),
                               d=normal(theta.d, pi_epsilon.d),
                               kappa=round(normal(theta.kappa, pi_epsilon.kappa)),
                               omega=round(normal(theta.omega, pi_epsilon.omega)))


def get_arguments() -> Namespace:
    """ Create the CLI and parse the arguments.

    :return: Namespace of all values.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description='ABC MCMC for microsatellite mutation model 4T1S2I parameter estimation.')

    list(map(lambda a: parser.add_argument(a[0], help=a[1], type=a[2], nargs=a[3], default=a[4]), [
        ['-odb', 'Location of the observed database file.', str, None, 'data/observed.db'],
        ['-mdb', 'Location of the database to record to.', str, None, 'data/ma1t0s0i.db'],
        ['-uid_observed', 'IDs of observed samples to compare to.', str, '+', None],
        ['-locus_observed', 'Loci of observed samples (must match with uid).', str, '+', None],
        ['-simulation_n', 'Number of simulations to use to obtain a distance.', int, None, None],
        ['-iterations_n', 'Number of iterations to run MCMC for.', int, None, None],
        ['-epsilon', "Maximum acceptance value for distance between [0, 1].", float, None, None],
        ['-flush_n', 'Number of iterations to run MCMC before flushing to disk.', int, None, None],
        ['-seed', '1 -> last recorded "mdb" position is used (TIME_R, PROPOSED_TIME).', None, None],
        ['-n_b', 'Sample size for common ancestor.', int, None, None],
        ['-n_s1', 'Sample size for intermediate 1.', int, None, None],
        ['-n_s2', 'Sample size for intermediate 2.', int, None, None],
        ['-n_e', 'Sample size for end population.', int, None, None],
        ['-f_b', 'Scaling factor for common ancestor mutation rate.', float, None, None],
        ['-f_s1', 'Scaling factor for intermediate 1 mutation rate.', float, None, None],
        ['-f_s2', 'Scaling factor for intermediate 2 mutation rate.', float, None, None],
        ['-f_e', 'Scaling factor for end population mutation rate.', float, None, None],
        ['-alpha', 'Admixture factor between S1 and S2, between [0, 1].', float, None, None],
        ['-c', 'Constant bias for the upward mutation rate.', float, None, None],
        ['-d', 'Linear bias for the downward mutation rate.', float, None, None],
        ['-kappa', 'Lower bound of repeat lengths.', int, None, None],
        ['-omega', 'Upper bound of repeat lengths.', int, None, None],
        ['-n_b_sigma', 'Step size of n_b when changing parameters.', float, None, None],
        ['-n_s1_sigma', 'Step size of n_s1 when changing parameters.', float, None, None],
        ['-n_s2_sigma', 'Step size of n_s2 when changing parameters.', float, None, None],
        ['-n_e_sigma', 'Step size of n_e when changing parameters.', float, None, None],
        ['-f_b_sigma', 'Step size of f_b when changing parameters.', float, None, None],
        ['-f_s1_sigma', 'Step size of f_s1 when changing parameters.', float, None, None],
        ['-f_s2_sigma', 'Step size of f_s2 when changing parameters.', float, None, None],
        ['-f_e_sigma', 'Step size of f_e when changing parameters.', float, None, None],
        ['-alpha_sigma', 'Step size of alpha when changing parameters', float, None, None],
        ['-c_sigma', 'Step size of c when changing parameters.', float, None, None],
        ['-d_sigma', 'Step size of d when changing parameters.', float, None, None],
        ['-kappa_sigma', 'Step size of kappa when changing parameters.', float, None, None],
        ['-omega_sigma', 'Step size of omega when changing parameters.', float, None, None]
    ]))

    return parser.parse_args()


if __name__ == '__main__':
    from sqlite3 import connect

    arguments = get_arguments()  # Parse our arguments.

    # Determine the arguments that aren't in a MCMC4T1S2I object.
    p_complement = ['odb', 'mdb', 'n_b', 'n_s1', 'n_s2', 'n_e', 'f_b', 'f_s1', 'f_s2', 'f_e', 'alpha', 'c', 'd',
                    'kappa', 'omega', 'n_b_sigma', 'n_s1_sigma', 'n_s2_sigma', 'n_e_sigma', 'f_b_sigma', 'f_s1_sigma',
                    'f_s2_sigma', 'f_e_sigma', 'alpha_sigma', 'c_sigma', 'd_sigma', 'kappa_sigma', 'omega_sigma']

    # Connect to all of our databases.
    connection_o, connection_m = connect(arguments.odb), connect(arguments.mdb)

    # Prepare an MCMC run (obtain frequencies, create tables).
    mcmc = MCMC4T1S2I(connection_m=connection_m, connection_o=connection_o,
                      theta_0=Parameter4T1S2I.from_namespace(arguments) if arguments.n is not None else None,
                      walk_params=Parameter4T1S2I.from_namespace(arguments, lambda a: a + '_sigma'),
                      **{k: v for k, v in vars(arguments).items() if k not in p_complement})

    # Run the MCMC.
    mcmc.run()

    # Close our connections.
    connection_m.commit(), connection_o.close(), connection_m.close()
