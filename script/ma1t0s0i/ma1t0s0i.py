#!/usr/bin/env python3
from numpy import nextafter, ndarray
from kumulaau import Parameters, Population
from kumulaau import Cosine, MCMCA
from argparse import Namespace
from typing import List


class Parameters1T0S0I(Parameters):
    PARAMETER_COUNT = 6  # Set our properties.

    def __init__(self, n: float, f: float, c: float, d: float, kappa: int, omega: int):
        """ Constructor.

        :param n: Population size, used for determining the number of generations between events.
        :param f: Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        """
        super().__init__(c, d, kappa, omega)
        self.n, self.f = round(n), f

    @staticmethod
    def _from_namespace(p) -> List:
        """ Return a list from a namespace in the same order of __iter__.

         :param p: Arguments from some namespace.
         :return: List from namespace.
         """
        return [p.n, p.f, p.c, p.d, p.kappa, p.omega]

    def _walk_criteria(self) -> bool:
        """ Determine if a current parameter set is valid.

        :return: True if valid. False otherwise.
        """
        return self.n > 0 and \
            self.f >= 0 and \
            self.c > 0 and \
            self.d >= 0 and \
            0 < self.kappa < self.omega


class Population1T0S0I(Population):
    def _transform_bounded(self, theta: Parameters1T0S0I) -> Parameters1T0S0I:
        """ Return a new parameter set, bounded by the constraints below.

        :param theta: Parameter1T0S0I set to transform.
        :return: A new Parameter1T0S0I set, properly bounded.
        """
        return Parameters1T0S0I(max(theta.n, 0),
                                max(theta.f, 0),
                                max(theta.c, nextafter(0, 1)),  # Dr. Reed gave a strict bound > 0 here, but I forget...
                                max(theta.d, 0),
                                max(theta.kappa, 0),
                                max(theta.omega, theta.kappa))

    def _trace_trees(self, theta: Parameters1T0S0I) -> List:
        """ Generate a 1-element list of pointers to a single tree in the pop module.

        :param theta: Parameter1T0S0I set to use with tree tracing.
        :return: 1-element list of pointers to pop module C structure.
        """
        return [self._pop_trace(theta.n, theta.f, theta.c, theta.d, theta.kappa, theta.omega)]

    def _resolve_lengths(self, i_0: ndarray) -> ndarray:
        """ Generate a list of lengths of our 1T (one total) 0S (zero splits) 0I (zero intermediates) model.

        :return: List of repeat lengths.
        """
        return self._pop_evolve(self.tree_pointers[0], i_0)


class MCMC1T0S0I(MCMCA):
    # Set the model name associated with the database.
    MODEL_NAME = "MA1T0S0I"

    # Set the model schema associated with the database.
    MODEL_SCHEME_SQL = "N INT, F FLOAT, C FLOAT, D FLOAT, KAPPA INT, OMEGA INT"

    # Set the parameter class to use.
    PARAMETER_CLASS = Parameters1T0S0I

    # Set the distance class to use.
    DISTANCE_CLASS = Cosine

    @staticmethod
    def _sample(theta, i_0) -> ndarray:
        """ Given some parameter set theta and an initial state i_0, return a population that represents sampling from
        the posterior distribution.

        :param theta: Current parameter set to sample.
        :param i_0: Common ancestor state.
        :return: Array of repeat lengths to compare with an observed population.
        """
        return Population1T0S0I(theta).evolve(i_0)

    @staticmethod
    def _walk(theta, pi_epsilon) -> Parameters1T0S0I:
        """ Given some parameter set theta and the sigma parameters pi_epsilon, generate a new parameter set.

        :param theta: Current point to walk from.
        :param pi_epsilon: Distribution parameters (deviation, with mean at theta) to walk with.
        :return: A new parameter set.
        """
        from numpy.random import normal

        return Parameters1T0S0I(n=round(normal(theta.n, pi_epsilon.n)),
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

    parser = ArgumentParser(description='ABC MCMC for microsatellite mutation model 1T0S0I parameter estimation.')

    list(map(lambda a: parser.add_argument(a[0], help=a[1], type=a[2], nargs=a[3], default=a[4]), [
        ['-odb', 'Location of the observed database file.', str, None, 'data/observed.db'],
        ['-mdb', 'Location of the database to record to.', str, None, 'data/ma1t0s0i.db'],
        ['-uid_observed', 'IDs of observed samples to compare to.', str, '+', None],
        ['-locus_observed', 'Loci of observed samples (must match with uid).', str, '+', None], 
        ['-simulation_n', 'Number of simulations to use to obtain a distance.', int, None, None],
        ['-iterations_n', 'Number of iterations to run MCMC for.', int, None, None],
        ['-epsilon', "Maximum acceptance value for distance between [0, 1].", float, None, None],
        ['-flush_n', 'Number of iterations to run MCMC before flushing to disk.', int, None, None],
        ['-n', 'Starting sample size (population size).', int, None, None],
        ['-f', 'Scaling factor for total mutation rate.', float, None, None],
        ['-c', 'Constant bias for the upward mutation rate.', float, None, None],
        ['-d', 'Linear bias for the downward mutation rate.', float, None, None],
        ['-kappa', 'Lower bound of repeat lengths.', int, None, None],
        ['-omega', 'Upper bound of repeat lengths.', int, None, None],
        ['-n_sigma', 'Step size of n when changing parameters.', float, None, None],
        ['-f_sigma', 'Step size of f when changing parameters.', float, None, None],
        ['-c_sigma', 'Step size of c when changing parameters.', float, None, None],
        ['-d_sigma', 'Step size of d when changing parameters.', float, None, None],
        ['-kappa_sigma', 'Step size of kappa when changing parameters.', float, None, None],
        ['-omega_sigma', 'Step size of omega when changing parameters.', float, None, None]
    ]))

    return parser.parse_args()


if __name__ == '__main__':
    from sqlite3 import connect

    arguments = get_arguments()  # Parse our arguments.

    # Determine the arguments that aren't in a MCMC1T0S0I object.
    p_complement = ['odb', 'mdb', 'n', 'f', 'c', 'd', 'kappa', 'omega', 'n_sigma', 'f_sigma', 'c_sigma',
                    'd_sigma', 'kappa_sigma', 'omega_sigma']

    # Connect to all of our databases.
    connection_o, connection_m = connect(arguments.odb), connect(arguments.mdb)

    # Prepare an MCMC run (obtain frequencies, create tables).
    mcmc = MCMC1T0S0I(connection_m=connection_m, connection_o=connection_o,
                      # theta_0=Parameters1T0S0I.from_args(arguments) if arguments.n is not None else None,
                      pi_epsilon=Parameters1T0S0I.from_args_sigma(arguments),
                      **{k: v for k, v in vars(arguments).items() if k not in p_complement})

    # Run the MCMC.
    mcmc.run()

    # Close our connections.
    connection_m.commit(), connection_o.close(), connection_m.close()
