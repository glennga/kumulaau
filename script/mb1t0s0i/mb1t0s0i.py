#!/usr/bin/env python3
from kumulaau.population import BaseParameters
from sqlite3 import Cursor
from typing import List, Callable


def create_tables(cursor: Cursor) -> None:
    """ Create the tables to log the results of our MCMC to.

    :param cursor: Cursor to the database file to log to.
    :return:    None.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS WASTELESS_OBSERVED (
            TIME_R TIMESTAMP,
            UID_OBSERVED TEXT,
            LOCUS_OBSERVED TEXT
        );""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS WASTELESS_MODEL (
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
        INSERT INTO WASTELESS_OBSERVED
        VALUES (?, ?, ?)
    """, ((date_string, a[0], a[1]) for a in zip(uid_observed, locus_observed)))

    # noinspection SqlInsertValues
    cursor.executemany(f"""
        INSERT INTO WASTELESS_MODEL
        VALUES ({','.join('?' for _ in range(x[0][0].PARAMETER_COUNT + 5))});
    """, ((date_string,) + tuple(a[0]) + (a[1], a[2], a[3], a[4]) for a in x))

    # Clear our chain except for the last state.
    x[:] = [x[-1]]


def retrieve_last(cursor: Cursor) -> BaseParameters:
    """ TODO:

    :param cursor:
    :return:
    """
    a = cursor.execute("""
        SELECT N, F, C, D, KAPPA, OMEGA
        FROM WASTELESS_MODEL
        ORDER BY TIME_R, PROPOSED_TIME DESC
        LIMIT 1
    """).fetchone()
    return BaseParameters(*a)


def compute_exact_likelihood(distances: ndarray):
    """ TODO: Finish this documentation.

    :param distances:
    :return:
    """
    from numpy import argsort, array, log, sqrt, linspace
    from numpy.linalg import lstsq
    from kumulaau.distance import Cosine

    # Obtain the CDF for our distances in descending order. We want this on the log scale.
    cdf, domain = argsort(-distances), array(range(distances.size)) / float(distances.size)
    cdf_log = log(cdf)

    # Our weight parameters. At complete_similarity, we are only accepting exact matches.
    w = linspace(Cosine().COMPLETE_MATCH, Cosine.COMPLETE_DIFFERENCE, cdf_log.size)

    # Perform our regression. We return the intercept, the probability of an exact match with these set of distances.
    return lstsq(cdf_log * sqrt(w), domain * sqrt(w))[0][1]


def mcmc(iterations_n: int, observed_frequencies: List, simulation_n: int,
        theta_0: BaseParameters, q_sigma: BaseParameters) -> List:
    """ A MCMC algorithm to approximate the posterior distribution of the mutation model, whose acceptance to the
    chain is determined by the distance between repeat length distributions. My interpretation of this new MCMC approach
    is given below:

    TODO: Fix this documentation below.
    1) We start with some initial guess, and simulate a population of size 'n'.
    2) For each observed frequency distribution...
        a) We compute a set of iterations_n distances.
        b) From this set of distances, we perform weighted linear regression

    2) We then compute the average distance between a set of simulated and observed samples.
        a) If this term is above some defined epsilon or ~U(0, 1) < A, we append this to our chain.
        b) Otherwise, we reject it and increase the waiting time of our current parameter state by one.
    3) Repeat for 'iterations_n' iterations.

    :param iterations_n: Number of iterations to run MCMC for.
    :param observed_frequencies: Dirty observed frequency samples.
    :param simulation_n: Number of simulations to use to obtain a distance.
    :param theta_0: Our initial guess for parameters.
    :param q_sigma: The deviations associated with all parameters to use when generating new parameters.
    :return: A chain of all states we visited (parameters), their associated waiting times, and the sum total of their
             acceptance probabilities.
    """
    from methoda import choose_i_0
    from population import Population
    from numpy.random import normal
    from numpy import nextafter
    from distance import Cosine

    # There exists two chains -> X: A Markov chain holding our accepted MCMC proposals. Y: The rejected states.
    x, y, l_prev = [[theta_0, 1, nextafter(0, 1), 0, '']], [], 0
    walk = lambda a, b: normal(a, b)  # Our proposal function is normally distributed.

    for iteration in range(1, iterations_n):
        theta_k, summary = x[-1][0], Cosine()
        theta_proposed = BaseParameters.from_walk(theta_k, q_sigma, walk)

        for _ in range(simulation_n):
            # Generate some population given the current parameter set.
            population = Population(theta_proposed).evolve(choose_i_0(observed_frequencies))
            summary.delta_rows(observed_frequencies, population)

        # Accept our proposal if delta > epsilon and if our acceptance probability condition is met.
        if acceptance(theta_proposed, theta_k):
            x = x + [[theta_proposed, 1, summary.average_distance(), iteration]]

        # Reject our proposal. We keep our current state, increment our waiting times, and add to the reject chain.
        else:
            x[-1][1] += 1
            y = y + [[theta_proposed, -1, summary.average_distance(), iteration]]

    return x[1:] + y  # We return **all** states explored.


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect

    parser = ArgumentParser(description='ABC MCMC for microsatellite mutation model parameter estimation.')
    parser.add_argument('-odb', help='Location of the observed database file.', type=str, default='data/observed.db')
    parser.add_argument('-mdb', help='Location of the database to record to.', type=str, default='data/method-a.db')
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
    main_theta_0 = BaseParameters.from_args(main_arguments, False) if main_arguments.seed != 1 else \
        retrieve_last(cursor_m)
    main_q_sigma = BaseParameters.from_args(main_arguments, True)
    main_log = lambda a, b: log_states(cursor_m, main_arguments.uid_observed, main_arguments.locus_observed, a) and \
        connection_m.commit() if b % main_arguments.flush_n == 0 else None

    main_iterations_start = 0 if main_arguments.seed == 0 else cursor_m.execute("""
        SELECT PROPOSED_TIME -- Determine our iteration boundaries. --
        FROM WASTELESS_MODEL
        ORDER BY PROPOSED_TIME DESC
        LIMIT 1
    """).fetchone()[0]
    main_iterations_bounds = [main_iterations_start, main_arguments.iterations_n + main_iterations_start + 1]

    # Perform the ABC MCMC. Record at every 'flush_n'.
    mcmc(main_iterations_bounds, main_observed_frequencies, main_arguments.simulation_n,
         main_arguments.epsilon, main_theta_0, main_q_sigma, main_log)

    # Remove the initial states of our chain.
    cursor_m.execute("""
        DELETE FROM WASTELESS_MODEL
        WHERE PROPOSED_TIME = 0
    """), connection_m.commit(), connection_o.close(), connection_m.close()