#!/usr/bin/env python3
from numpy import nextafter, ndarray
from kumulaau import Parameters, Population
from typing import List, Callable
from argparse import Namespace
from sqlite3 import Cursor


class Parameters1T0S0I(Parameters):
    def __init__(self, n: float, f: float, c: float, d: float, kappa: int, omega: int):
        """ Constructor. This is just meant to be a data class for the mutation model. Note that if any changes are made
        to the quantity or type of parameters here, they MUST be changed in "_pop.c" as well.

        :param n: Population size, used for determining the number of generations between events.
        :param f: Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        """
        super().__init__(c, d, kappa, omega)
        self.n, self.f = round(n), f

    def PARAMETER_COUNT(self):
        """ The number of parameters we have.

        :return: The number of parameters we have.
        """
        return 6

    @staticmethod
    def _from_namespace(p) -> List:
        """ Return a list from a namespace in the same order of __iter__.

         :param p: Arguments from some namespace.
         :return: List from namespace.
         """
        return [p.n, p.f, p.c, p.d, p.kappa, p.omega]

    @staticmethod
    def _sigma_namespace(p: Namespace) -> List:
        """ Return a list of '_sigma' values from a namespace in the same order of the constructor.

        :param p: Arguments from some namespace.
        :return: List from namespace.
        """
        return [p.n_sigma, p.f_sigma, p.c_sigma, p.d_sigma, p.kappa_sigma, p.omega_sigma]

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


def create_tables(cursor: Cursor) -> None:
    """ Create the tables to log the results of our MCMC to.

    :param cursor: Cursor to the database file to log to.
    :return: None.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MA1T0S0I_OBSERVED (
            TIME_R TIMESTAMP,
            UID_OBSERVED TEXT,
            LOCUS_OBSERVED TEXT
        );""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MA1T0S0I_MODEL (
            TIME_R TIMESTAMP,
            N INT,
            F FLOAT,
            C FLOAT,
            D FLOAT,
            KAPPA INT,
            OMEGA INT,
            WAITING_TIME INT,
            LIKELIHOOD FLOAT,
            DISTANCE FLOAT,
            PROPOSED_TIME INT
        );""")


def log_states(cursor: Cursor, uid_observed: List[str], locus_observed: List[str], x: List) -> None:
    """ Record our states to some database.

    :param cursor: Cursor to the database file to log to.
    :param uid_observed: IDs of the observed samples to compare to.
    :param locus_observed: Loci of the observed samples to compare to.
    :param x: States and associated times & probabilities collected after running MCMC (our Markov chain).
    :return: None.
    """
    from datetime import datetime
    date_string = datetime.now()

    if len(x) == 0:  # We can't record if we have nothing to record!
        return

    # Record our observed sample log strings and datetime.
    cursor.executemany("""
        INSERT INTO MA1T0S0I_OBSERVED
        VALUES (?, ?, ?)
    """, ((date_string, a[0], a[1]) for a in zip(uid_observed, locus_observed)))

    # noinspection SqlInsertValues
    cursor.executemany(f"""
        INSERT INTO MA1T0S0I_MODEL
        VALUES ({','.join('?' for _ in range(x[0][0].PARAMETER_COUNT + 5))});
    """, ((date_string,) + tuple(a[0]) + (a[1], a[2], a[3], a[4]) for a in x))

    # Clear our chain except for the last state.
    x[:] = [x[-1]]


def retrieve_last(cursor: Cursor) -> Parameters1T0S0I:
    """ Retrieve the last parameter set from a previous run. This is meant to be used for continuing MCMC runs.

    :param cursor: Cursor to the database file holding the results of a previous run.
    :return: BaseParameters holding the parameter set from last recorded run.
    """
    a = cursor.execute("""
        SELECT N, F, C, D, KAPPA, OMEGA
        FROM MA1T0S0I_MODEL
        ORDER BY TIME_R, PROPOSED_TIME DESC
        LIMIT 1
    """).fetchone()
    return Parameters1T0S0I(*a)


def mcmc(iterations_bounds: List, observed_frequencies: List, simulation_n: int,
         epsilon: float, theta_0: Parameters1T0S0I, q_sigma: Parameters1T0S0I, log: Callable) -> None:
    """ A MCMC algorithm to approximate the posterior distribution of the mutation model, whose acceptance to the
    chain is determined by the distance between repeat length distributions. My interpretation of this ABC-MCMC approach
    is given below:

    1) We start with some initial guess theta_0. Right off the bat, we move to another theta from theta_0.
    2) For 'iterations_bounds[1] - iterations_bounds[0]' iterations...
        a) For 'simulation_n' iterations...
            i) We simulate a population using the given theta.
            ii) For each observed frequency ... 'D'
                1) We compute the difference between the two distributions.
                2) If this difference is less than our epsilon term, we add 1 to a vector modeling D.
        b) Compute the probability that each observed frequency matches a generated population: all of D / simulation_n.
        c) If this probability is greater than the probability of the previous, we accept.
        d) Otherwise, we accept our proposed with probability p(proposed) / p(prev).

    :param iterations_bounds: Number of iterations to run MCMC between.
    :param observed_frequencies: Dirty observed frequency samples.
    :param simulation_n: Number of simulations to use to obtain a distance.
    :param epsilon: Minimum acceptance value for our distance.
    :param theta_0: Our initial guess for parameters.
    :param q_sigma: The deviations associated with all parameters to use when generating new parameters.
    :param log: A log function, used to flush our chain to disk.
    :return: None.
    """
    from kumulaau.population import Population
    from numpy.random import normal, uniform
    from kumulaau.distance import Cosine

    x = [[theta_0, 1, 1.0e-10, 1.0e-10, 0]]  # Seed our Markov chain with our initial guess.
    sample, walk = lambda a, b: Population(a).evolve(b), lambda a, b: normal(a, b)

    for i in range(iterations_bounds[0] + 1, iterations_bounds[1]):  # Walk from our previous state.
        theta_proposed, theta_k = Parameters1T0S0I.from_walk(x[-1][0], q_sigma, walk), x[-1][0]
        delta = Cosine(observed_frequencies, theta_proposed.kappa, theta_proposed.omega, simulation_n)

        # Compute our matched and delta matrix (simulation (rows) by observation (columns)). Get a mean distance.
        expected_distance = delta.fill_matrices(sample, theta_proposed, epsilon)

        # Accept our proposal according to our alpha value.
        p_proposed, p_k = delta.match_likelihood(), x[-1][2]
        if p_proposed / p_k > uniform(0, 1):
            x = x + [[theta_proposed, 1, p_proposed, expected_distance, i]]

        # Reject our proposal. We keep our current state and increment our waiting times.
        else:
            x[-1][1] += 1

        # We record to our chain. This is dependent on the current iteration of MCMC.
        log(x, i)


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    from sqlite3 import connect

    parser = ArgumentParser(description='ABC MCMC for microsatellite mutation model parameter estimation (1 pop).')
    parser.add_argument('-odb', help='Location of the observed database file.', type=str, default='data/observed.db')
    parser.add_argument('-mdb', help='Location of the database to record to.', type=str, default='data/ma1t0s0i.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    parser.add_argument('-uid_observed', help='IDs of observed samples to compare to.', type=str, nargs='+')
    parser.add_argument('-locus_observed', help='Loci of observed samples (must match with uid).', type=str, nargs='+')
    paa('-simulation_n', 'Number of simulations to use to obtain a distance.', int)
    paa('-iterations_n', 'Number of iterations to run MCMC for.', int)
    paa('-epsilon', "Maximum acceptance value for distance between [0, 1].", float)

    paa('-flush_n', 'Number of iterations to run MCMC before flushing to disk.', int)
    paa('-seed', '1 -> last recorded "mdb" position is used (TIME_R, PROPOSED_TIME).', int)

    paa('-n', 'Starting sample size (population size).', int)
    paa('-f', 'Scaling factor for total mutation rate.', float)
    paa('-c', 'Constant bias for the upward mutation rate.', float)
    paa('-d', 'Linear bias for the downward mutation rate.', float)
    paa('-kappa', 'Lower bound of repeat lengths.', int)
    paa('-omega', 'Upper bound of repeat lengths.', int)

    paa('-n_sigma', 'Step size of n when changing parameters.', float)
    paa('-f_sigma', 'Step size of f when changing parameters.', float)
    paa('-c_sigma', 'Step size of c when changing parameters.', float)
    paa('-d_sigma', 'Step size of d when changing parameters.', float)
    paa('-kappa_sigma', 'Step size of kappa when changing parameters.', float)
    paa('-omega_sigma', 'Step size of omega when changing parameters.', float)
    main_arguments = parser.parse_args()  # Parse our arguments.

    # Connect to all of our databases.
    connection_o, connection_m = connect(main_arguments.odb), connect(main_arguments.mdb)
    cursor_o, cursor_m = connection_o.cursor(), connection_m.cursor()
    create_tables(cursor_m)

    main_observed_frequencies = list(map(lambda a, b: cursor_o.execute(""" -- Get frequencies from observed database. --
        SELECT ELL, ELL_FREQ
        FROM OBSERVED_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (a, b,)).fetchall(), main_arguments.uid_observed, main_arguments.locus_observed))

    # Parse our starting point. If 'seed' is specified, we use this over any (n, f, c, d, u, ...) given.
    main_theta_0 = Parameters1T0S0I.from_args(main_arguments, False) if main_arguments.seed != 1 else \
        retrieve_last(cursor_m)
    main_q_sigma = Parameters1T0S0I.from_args(main_arguments, True)
    main_log = lambda a, b: log_states(cursor_m, main_arguments.uid_observed, main_arguments.locus_observed, a) and \
        connection_m.commit() if b % main_arguments.flush_n == 0 else None

    main_iterations_start = 0 if main_arguments.seed == 0 else cursor_m.execute("""
        SELECT PROPOSED_TIME -- Determine our iteration boundaries. --
        FROM MA1T0S0I_MODEL
        ORDER BY PROPOSED_TIME DESC
        LIMIT 1
    """).fetchone()[0]
    main_iterations_bounds = [main_iterations_start, main_arguments.iterations_n + main_iterations_start + 1]

    # Perform the ABC MCMC. Record at every 'flush_n'.
    mcmc(main_iterations_bounds, main_observed_frequencies, main_arguments.simulation_n,
         main_arguments.epsilon, main_theta_0, main_q_sigma, main_log)

    # Remove the initial states of our chain.
    cursor_m.execute("""
        DELETE FROM MA1T0S0I_MODEL
        WHERE PROPOSED_TIME = 0
    """), connection_m.commit(), connection_o.close(), connection_m.close()
