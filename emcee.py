#!/usr/bin/env python3
from single import ModelParameters
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
            REAL_SAMPLE_UID TEXT,
            REAL_LOCUS TEXT,
            I_0 TEXT,
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
            ACCEPTANCE_TIME
        );""")


def log_states(cur_j: Cursor, rsu: str, l: str, chain: List) -> None:
    """ Record our states to some database.

    :param cur_j: Cursor to the database file to log to.
    :param rsu: ID of the real sample data set to compare to.
    :param l: Locus of the real sample to compare to.
    :param chain: States and associated times & probabilities collected after running MCMC.
    :return: None.
    """
    from datetime import datetime

    for state in chain:
        cur_j.execute(f"""
            INSERT INTO WAIT_POP
            VALUES ({','.join('?' for _ in range(16))});
        """, (datetime.now(), rsu, l, '-'.join(str(a) for a in state[0].i_0), state[0].big_n, state[0].mu,
              state[0].s, state[0].kappa, state[0].omega, state[0].u, state[0].v, state[0].m, state[0].p,
              state[1], state[2], state[3]))


def metro_hast(it: int, rfs: List, r: int, two_n: int, epsilon: int, parameters_init: ModelParameters,
               parameters_sigma: ModelParameters) -> List:
    """ My interpretation of the Metropolis-Hastings algorithm w/ ABC. We start with some initial guess and compute the
    acceptance probability of these current parameters. We generate another proposal by walking randomly in our
    parameter space to another parameter set, and determine the data difference (delta) here. If this is greater than
    some defined epsilon, we accept this new parameter set. If we deny our proposal, we increment our waiting time
    and try again. Sources:

    https://advaitsarkar.wordpress.com/2014/03/02/the-metropolis-hastings-algorithm-tutorial/
    https://theoreticalecology.wordpress.com/2012/07/15/a-simple-approximate-bayesian-computation-mcmc-abc-mcmc-in-r/

    :param it: Number of iterations to run MCMC for.
    :param rfs: Real frequency sample. First column is the length, second is the frequency.
    :param r: Number of populations to use to obtain delta.
    :param two_n: Sample size of the alleles. Must match the real sample given here.
    :param epsilon: Minimum acceptance value for delta.
    :param parameters_init: Our initial guess for parameters.
    :param parameters_sigma: The deviations associated with all parameters to use when generating new parameters.
    :return: A chain of all states we visited (parameters), their associated waiting times, and the sum total of their
             acceptance probabilities.
    """
    from numpy.random import normal
    from numpy import average, array
    from single import Single
    from compare import compare

    # Seed our chain with our initial guess.
    states = [[parameters_init, 1, 1 - average(compare(r, two_n, rfs, Single(parameters_init).evolve())), 0]]

    # Determine how we walk across our parameter space.
    walk = lambda a, b: normal(a, b)
    walk_i_0 = lambda a: array([round(walk(aa, parameters_sigma.i_0)) for aa in a])

    for j in range(1, it):
        parameters_prev = states[-1][0]  # Our current position in the state space. Walk from this point.
        parameters_proposed = ModelParameters(i_0=walk_i_0(parameters_prev.i_0),
                                              big_n=round(walk(parameters_prev.big_n, parameters_sigma.big_n)),
                                              mu=walk(parameters_prev.mu, parameters_sigma.mu),
                                              s=walk(parameters_prev.s, parameters_sigma.s),
                                              kappa=round(walk(parameters_prev.kappa, parameters_sigma.kappa)),
                                              omega=round(walk(parameters_prev.omega, parameters_sigma.omega)),
                                              u=walk(parameters_prev.u, parameters_sigma.u),
                                              v=walk(parameters_prev.v, parameters_sigma.v),
                                              m=walk(parameters_prev.m, parameters_sigma.m),
                                              p=walk(parameters_prev.p, parameters_sigma.p))
        
        # Generate and evolve a single population given the proposed parameters. Compute the delta.
        summary_delta = 1 - average(compare(r, two_n, rfs, Single(parameters_proposed).evolve()))

        # Accept our proposal if the current delta is above some defined epsilon.
        if summary_delta > epsilon:
            states = states + [[parameters_proposed, 1, summary_delta, j]]

        # Reject our proposal. We keep our current state and increment our waiting times.
        else:
            states[-1][1] += 1

    return states


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect

    parser = ArgumentParser(description='Evolve allele populations with different parameter sets using a grid search.')
    parser.add_argument('-rdb', help='Location of the real database file.', type=str, default='data/real.db')
    parser.add_argument('-edb', help='Location of the database to record to.', type=str, default='data/emcee.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)
    
    paa('-r', 'Number of populations to use to obtain delta.', int)
    paa('-epsilon', 'Minimum acceptance value for delta.', float)
    paa('-it', 'Number of iterations to run MCMC for.', int)
    paa('-rsu', 'ID of the real sample data set to compare to.', str)
    paa('-l', 'Locus of the real sample to compare to.', str)

    parser.add_argument('-i_0', help='Repeat lengths of starting ancestors.', type=int, nargs='+')
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
    paa('-p_sigma', 'Step size of p when changing parameters.', float)
    args = parser.parse_args()  # Parse our arguments.

    # Connect to all of our databases.
    conn_r, conn_e = connect(args.rdb), connect(args.edb)
    cur_r, cur_e = conn_r.cursor(), conn_e.cursor()
    create_table(cur_e)

    freq_r = cur_r.execute(""" -- Pull the frequency distribution from the real database. --
        SELECT ELL, ELL_FREQ
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (args.rsu, args.l, )).fetchall()

    two_nm = int(cur_r.execute(""" -- Retrieve the sample size, the number of alleles. --
        SELECT SAMPLE_SIZE
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (args.rsu, args.l, )).fetchone()[0])

    # Perform the MCMC, and record our chain.
    log_states(cur_e, args.rsu, args.l, metro_hast(args.it, freq_r, args.r, two_nm, args.epsilon,
                                                   ModelParameters(i_0=args.i_0, big_n=args.big_n, mu=args.mu,
                                                                   s=args.s, kappa=args.kappa, omega=args.omega,
                                                                   u=args.u, v=args.v, m=args.m, p=args.p),
                                                   ModelParameters(i_0=args.i_0_sigma, big_n=args.big_n_sigma,
                                                                   mu=args.mu_sigma, s=args.s_sigma,
                                                                   kappa=args.kappa_sigma, omega=args.omega_sigma,
                                                                   u=args.u_sigma, v=args.v_sigma, m=args.m_sigma,
                                                                   p=args.p_sigma)))
    conn_e.commit()
