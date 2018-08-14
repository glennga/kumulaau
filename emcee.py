#!/usr/bin/env python3
from single import ModelParameters
from numpy import ndarray
from sqlite3 import Cursor
from typing import List


def create_table(cur_j: Cursor) -> None:
    """ Create the table to log the results of our comparisons to.

    :param cur_j: Cursor to the database file to log to.
    :return: None.
    """
    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_POP (
            TIME_R TIMESTAMP,
            REAL_SAMPLE_UIDS TEXT,
            REAL_LOCI TEXT,
            BIG_N INT,
            MU FLOAT,
            S FLOAT,
            KAPPA INT,
            OMEGA INT,
            U FLOAT,
            V FLOAT,
            M FLOAT,
            P FLOAT,
            WAITING INT,
            DELTA FLOAT,
            ACCEPTANCE_TIME INT
        );""")


def log_states(cur_j: Cursor, rsu: List[str], l: List[str], chain: List) -> None:
    """ Record our states to some database.

    :param cur_j: Cursor to the database file to log to.
    :param rsu: IDs of the real sample data sets to compare to.
    :param l: Loci of the real samples to compare to.
    :param chain: States and associated times & probabilities collected after running MCMC.
    :return: None.
    """
    from datetime import datetime

    # Generate our real sample log strings.
    rsu_log, l_log = list(map(lambda a: '-'.join(b for b in a), [rsu, l]))

    for state in chain:
        cur_j.execute(f"""
            INSERT INTO WAIT_POP
            VALUES ({','.join('?' for _ in range(15))});
        """, (datetime.now(), rsu_log, l_log, state[0].big_n, state[0].mu,
              state[0].s, state[0].kappa, state[0].omega, state[0].u, state[0].v, state[0].m, state[0].p,
              state[1], state[2], state[3]))


def choose_i_0(rfs: List) -> ndarray:
    """ We treat the starting population length ancestor as a nuisance parameter. We randomly choose a repeat length
    from our real samples.

    :param rfs: Real frequency samples. A list of: [first column is the length, second is the frequency].
    :return: A single repeat length, wrapped in a Numpy array.
    """
    from random import choice
    from numpy import array

    return array([int(choice(choice(rfs))[0])])


def metro_hast(it: int, rfs_d: List, r: int, two_n: List[int], epsilon: float, theta_init: ModelParameters,
               theta_sigma: ModelParameters) -> List:
    """ My interpretation of an MCMC approach with ABC. We start with some initial guess and compute the
    acceptance probability of these current parameters. We generate another proposal by walking randomly in our
    parameter space to another parameter set, and determine the data difference (delta) here. If this is greater than
    some defined epsilon, we accept this new parameter set. If we deny our proposal, we increment our waiting time
    and try again. Sources:

    https://advaitsarkar.wordpress.com/2014/03/02/the-metropolis-hastings-algorithm-tutorial/
    https://theoreticalecology.wordpress.com/2012/07/15/a-simple-approximate-bayesian-computation-mcmc-abc-mcmc-in-r/

    :param it: Number of iterations to run MCMC for.
    :param rfs_d: Dirty real frequency samples. A list of: [first column is the length, second is the frequency].
    :param r: Number of simulated samples to use to obtain delta.
    :param two_n: Sample sizes of the alleles. Must match the order of the real samples given here.
    :param epsilon: Minimum acceptance value for delta.
    :param theta_init: Our initial guess for parameters.
    :param theta_sigma: The deviations associated with all parameters to use when generating new parameters.
    :return: A chain of all states we visited (parameters), their associated waiting times, and the sum total of their
             acceptance probabilities.
    """
    from numpy.random import normal
    from numpy import average
    from single import Single
    from compare import compare, prepare_compare

    states = [[theta_init, 1, 0, 0]]  # Seed our chain with our initial guess.
    walk = lambda a, b, c=False: normal(a, b) if c is False else round(normal(a, b))

    for j in range(1, it):
        theta_prev = states[-1][0]  # Our current position in the state space. Walk from this point.
        theta_proposed = ModelParameters(i_0=choose_i_0(rfs_d), big_n=walk(theta_prev.big_n, theta_sigma.big_n, True),
                                         mu=walk(theta_prev.mu, theta_sigma.mu), s=walk(theta_prev.s, theta_sigma.s),
                                         kappa=walk(theta_prev.kappa, theta_sigma.kappa, True),
                                         omega=walk(theta_prev.omega, theta_sigma.omega, True),
                                         u=walk(theta_prev.u, theta_sigma.u), v=walk(theta_prev.v, theta_sigma.v),
                                         m=walk(theta_prev.m, theta_sigma.m), p=walk(theta_prev.p, theta_sigma.p))

        # Generate some population given the current parameter set.
        z = Single(theta_proposed).evolve()

        summary_deltas = []  # Compute the delta term: the average probability using all available samples.
        for d in zip(rfs_d, two_n):
            scs, sfs, rfs, delta_rs = prepare_compare(d[0], z, d[1], r)
            compare(scs, sfs, rfs, z, delta_rs)  # Messy, but so is Numba. ):<
            summary_deltas = summary_deltas + [1 - average(delta_rs)]
        summary_delta = average(summary_deltas)

        # Accept our proposal if the current delta is above some defined epsilon (ABC).
        if summary_delta > epsilon:  # TODO: Record the generated population as well??
            states = states + [[theta_proposed, 1, summary_delta, j]]

        # Reject our proposal. We keep our current state and increment our waiting times.
        else:
            states[-1][1] += 1

    return states[1:]


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect
    from numpy import array

    parser = ArgumentParser(description='MCMC with ABC for microsatellite mutation parameter estimation.')
    parser.add_argument('-rdb', help='Location of the real database file.', type=str, default='data/real.db')
    parser.add_argument('-edb', help='Location of the database to record to.', type=str, default='data/emcee.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    parser.add_argument('-rsu', help='IDs of real samples to compare to.', type=str, nargs='+')
    parser.add_argument('-l', help='Loci of real samples to compare to (must match with rsu).', type=str, nargs='+')
    paa('-r', 'Number of simulated samples to use to obtain delta.', int)
    paa('-epsilon', 'Minimum acceptance value for delta.', float)
    paa('-it', 'Number of iterations to run MCMC for.', int)

    paa('-big_n', 'Starting effective population size.', int)
    paa('-mu', 'Starting mutation rate, bounded by (0, infinity).', float)
    paa('-s', 'Starting proportional rate, bounded by (-1 / (omega - kappa + 1), infinity).', float)
    paa('-kappa', 'Starting lower bound of possible repeat lengths.', int)
    paa('-omega', 'Starting upper bounds of possible repeat lengths.', int)
    paa('-u', 'Starting constant bias parameter, bounded by [0, 1].', float)
    paa('-v', 'Starting linear bias parameter, bounded by (-infinity, infinity).', float)
    paa('-m', 'Starting success probability for truncated geometric distribution.', float)
    paa('-p', 'Starting probability that the repeat length change is +/- 1.', float)

    paa('-i_0_sigma', 'Step size of i_0 when changing parameters.', float)
    paa('-big_n_sigma', 'Step size of big_n when changing parameters.', float)
    paa('-mu_sigma', 'Step size of mu when changing parameters.', float)
    paa('-s_sigma', 'Step size of s when changing parameters.', float)
    paa('-kappa_sigma', 'Step size of kappa when changing parameters.', float)
    paa('-omega_sigma', 'Step size of omega when changing parameters.', float)
    paa('-u_sigma', 'Step size of u when changing parameters.', float)
    paa('-v_sigma', 'Step size of v when changing parameters.', float)
    paa('-m_sigma', 'Step size of m when changing parameters.', float)
    paa('-p_sigma', 'Step size of p when changing theta.', float)
    args = parser.parse_args()  # Parse our arguments.

    # Connect to all of our databases.
    conn_r, conn_e = connect(args.rdb), connect(args.edb)
    cur_r, cur_e = conn_r.cursor(), conn_e.cursor()
    create_table(cur_e)

    freq_r = list(map(lambda a, b: cur_r.execute(""" -- Pull the frequency distributions from the real database. --
        SELECT ELL, ELL_FREQ
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
      """, (a, b,)).fetchall(), args.rsu, args.l))

    two_nm = list(map(lambda a, b: int(cur_r.execute(""" -- Retrieve the sample sizes, the number of alleles. --
        SELECT SAMPLE_SIZE
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (a, b,)).fetchone()[0]), args.rsu, args.l))

    # Perform the MCMC, and record our chain.
    log_states(cur_e, args.rsu, args.l, metro_hast(args.it, freq_r, args.r, two_nm, args.epsilon,
                                                   ModelParameters(i_0=choose_i_0(freq_r), big_n=args.big_n,
                                                                   mu=args.mu, s=args.s, kappa=args.kappa,
                                                                   omega=args.omega, u=args.u, v=args.v, m=args.m,
                                                                   p=args.p),
                                                   ModelParameters(i_0=choose_i_0(freq_r), big_n=args.big_n_sigma,
                                                                   mu=args.mu_sigma, s=args.s_sigma,
                                                                   kappa=args.kappa_sigma, omega=args.omega_sigma,
                                                                   u=args.u_sigma, v=args.v_sigma, m=args.m_sigma,
                                                                   p=args.p_sigma)))
    conn_e.commit()
