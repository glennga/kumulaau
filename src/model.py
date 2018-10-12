#!/usr/bin/env python3
from population import BaseParameters
from numpy import ndarray
from sqlite3 import Cursor
from typing import List, Tuple, Callable


def create_tables(cursor: Cursor) -> None:
    """ Create the tables to log the results of our MCMC to.

    :param cursor: Cursor to the database file to log to.
    :return: None.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_OBSERVED (
            TIME_R TIMESTAMP,
            UID_OBSERVED TEXT,
            LOCUS_OBSERVED TEXT
        );""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_MODEL (
            TIME_R TIMESTAMP,
            N INT,
            F FLOAT,
            C FLOAT,
            U FLOAT,
            D FLOAT,
            KAPPA INT,
            OMEGA INT,
            WAITING_TIME INT,
            DISTANCE FLOAT,
            ACCEPTANCE_TIME INT
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

    # Record our observed sample log strings and datetime.
    cursor.executemany("""
        INSERT INTO WAIT_OBSERVED
        VALUES (?, ?, ?)
    """, ((date_string, a[0], a[1]) for a in zip(uid_observed, locus_observed)))

    cursor.executemany(f"""
        INSERT INTO WAIT_MODEL
        VALUES ({','.join('?' for _ in range(x[0][0].PARAMETER_COUNT + 4))});
    """, ((date_string,) + tuple(a[0]) + (a[1], a[2], a[3]) for a in x))


def choose_i_0(observed_frequencies: List) -> ndarray:
    """ We treat the starting repeat length ancestor as a nuisance parameter. We randomly choose a repeat length
    from our observed samples.

    :param observed_frequencies: Observed frequency samples.
    :return: A single repeat length, wrapped in a Numpy array.
    """
    from random import choice
    from numpy import array

    return array([int(choice(observed_frequencies)[0])])


def condense_frequencies(observed_frequencies: List, n_hats: List[int]) -> Tuple[List, int]:
    """ TODO: Finish this documentation.

    :param observed_frequencies:
    :param n_hats:
    :return:
    """
    from collections import Counter

    # Generate a population of repeat lengths given sets of frequencies and sample sizes.
    population = []
    for frequencies in zip(observed_frequencies, n_hats):
        for repeat_unit in frequencies[0]:
            population = [repeat_unit[0] for _ in range(round(repeat_unit[1] * frequencies[1]))] + population

    # Determine the total frequency of each repeat length for this new population
    population_counter, total_n_hat = Counter(population), sum(n_hats)
    total_population_frequencies = [(a[0], a[1] / total_n_hat) for a in population_counter.items()]

    return total_population_frequencies, total_n_hat


def accept_theta_proposed(theta_proposed: BaseParameters, theta_k: BaseParameters):
    """ TODO: Finish acceptance_probability for documentation.

    :param theta_proposed:
    :param theta_k:
    :return:
    """
    from numpy.random import uniform
    from scipy.stats import beta
    from numpy import nextafter

    # We assume a beta prior for the following variables, and a uniform prior for the rest.
    beta_c = lambda a: beta.pdf(a, 1.48949, 4.14753, -1.71507e-05, 0.00618)
    beta_u = lambda a: beta.pdf(a, 7.40276, 4.62949, 1.17313, 0.04163)
    beta_d = lambda a: beta.pdf(a, 2.23564, 4.97695, -4.51903e-05, 0.00243)
    r = uniform(0, 1)

    # Evaluate our prior.
    beta_prior = (beta_c(theta_proposed.c) / beta_c(theta_k.c)) < r and \
                 (beta_u(theta_proposed.u) / beta_u(theta_k.u)) < r and \
                 (beta_d(theta_proposed.d) / beta_d(theta_k.d)) < r

    # Our proposal is symmetric (Metropolis algorithm). We perform bounds checking with our prior.
    return theta_proposed.n > 0 and \
        theta_proposed.f > 0 and \
        theta_proposed.c > nextafter(0, 1) and \
        theta_proposed.u > 1 and \
        theta_proposed.d > 0 and \
        0 < theta_proposed.kappa < theta_proposed.omega and \
        beta_prior


def mcmc(iterations_n: int, observed_frequency: List, sample_n: int, population_n: int, n_hat: int,
         epsilon: float, theta_0: BaseParameters, q_sigma: BaseParameters) -> List:
    """ A MCMC algorithm to approximate the posterior distribution of the mutation model, whose acceptance to the
    chain is determined by the distance between repeat length distributions. My interpretation of this ABC-MCMC approach
    is given below:

    1) We start with some initial guess and simulate an entire population.
    2) We then compute the average distance between a set of simulated and observed samples.
        a) If this term is above some defined epsilon or ~U(0, 1) < A, we append this to our chain.
        b) Otherwise, we reject it and increase the waiting time of our current parameter state by one.
    3) Repeat for 'iterations_n' iterations.

    :param iterations_n: Number of iterations to run MCMC for.
    :param observed_frequency: Dirty observed frequency sample.
    :param sample_n: Number of samples per simulation to use to obtain a distance.
    :param population_n: Number of simulations to use to obtain a distance.
    :param n_hat: Sample size of the alleles.
    :param epsilon: Minimum acceptance value for our distance.
    :param theta_0: Our initial guess for parameters.
    :param q_sigma: The deviations associated with all parameters to use when generating new parameters.
    :return: A chain of all states we visited (parameters), their associated waiting times, and the sum total of their
             acceptance probabilities.
    """
    from population import Population
    from numpy.random import normal
    from numpy import nextafter
    from summary import Cosine

    x, summary = [[theta_0, 1, nextafter(0, 1), 0, '']], None  # Seed our Markov chain with our initial guess.
    walk = lambda a, b: normal(a, b)

    for iteration in range(1, iterations_n):
        theta_k = x[-1][0]  # Our current position in the state space. Walk from this point.
        theta_proposed = BaseParameters.from_walk(theta_k, q_sigma, walk)

        for _ in range(population_n):
            # Generate some population given the current parameter set.
            population = Population(theta_proposed).evolve(choose_i_0(observed_frequency))
            summary = Cosine(population, 0, sample_n)

            summary.n_hat = n_hat  # Compute the distance term.
            summary.compute_distance(observed_frequency)

        # Accept our proposal if delta > epsilon and if our acceptance probability condition is met.
        if summary.average_distance() > epsilon and accept_theta_proposed(theta_proposed):
            x = x + [[theta_proposed, 1, summary.average_distance(), iteration]]

        # Reject our proposal. We keep our current state and increment our waiting times.
        else:
            x[-1][1] += 1

    return x[1:]


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect
    from numpy import array

    parser = ArgumentParser(description='MCMC for the *microsatellite mutation model* parameter estimation.')
    parser.add_argument('-odb', help='Location of the observed database file.', type=str, default='data/observed.db')
    parser.add_argument('-mdb', help='Location of the database to record to.', type=str, default='data/model.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    parser.add_argument('-uid_observed', help='IDs of observed samples to compare to.', type=str, nargs='+')
    parser.add_argument('-locus_observed', help='Loci of observed samples (must match with uid).', type=str, nargs='+')
    paa('-simulation_n', 'Number of simulations to use to obtain a distance.', int)
    paa('-sample_n', 'Number of samples per simulation to use to obtain a distance.', int)
    paa('-epsilon', 'Minimum acceptance value for distance between summary statistic.', float)
    paa('-iterations_n', 'Number of iterations to run MCMC for.', int)

    paa('-n', 'Starting effective population size.', int)
    paa('-f', 'Scaling factor for total mutation rate.', float)
    paa('-c', 'Constant bias for the upward mutation rate.', float)
    paa('-u', 'Linear bias for the upward mutation rate.', float)
    paa('-d', 'Linear bias for the downward mutation rate.', float)
    paa('-kappa', 'Lower bound of repeat lengths.', int)
    paa('-omega', 'Upper bound of repeat lengths.', int)

    paa('-n_sigma', 'Step size of big_n when changing parameters.', float)
    paa('-f_sigma', 'Step size of f when changing parameters.', float)
    paa('-c_sigma', 'Step size of c when changing parameters.', float)
    paa('-u_sigma', 'Step size of u when changing parameters.', float)
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

    main_n_hats = list(map(lambda a, b: int(cursor_o.execute(""" -- Retrieve the sample sizes, the number of alleles. --
        SELECT SAMPLE_SIZE
        FROM OBSERVED_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (a, b,)).fetchone()[0]), main_arguments.uid_observed, main_arguments.locus_observed))

    # Convert our collection of observations into one list.
    main_observed_frequency, main_h_hat = condense_frequencies(main_observed_frequencies, main_n_hats)

    # Perform the MCMC, and record our chain.
    main_theta_0 = BaseParameters.from_args(main_arguments, False)
    main_q_sigma = BaseParameters.from_args(main_arguments, True)
    log_states(cursor_m, main_arguments.uid_observed, main_arguments.locus_observed,
               mcmc(main_arguments.iterations_n, main_observed_frequency, main_arguments.sample_n,
                    main_arguments.simulation_n, main_h_hat, main_arguments.epsilon, main_theta_0, main_q_sigma))

    connection_m.commit(), connection_o.close(), connection_m.close()
