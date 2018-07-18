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
            I_0 INT,
            BIG_N INT,
            MU FLOAT,
            S FLOAT,
            KAPPA INT,
            OMEGA INT,
            U FLOAT,
            V FLOAT,
            M FLOAT,
            P FLOAT,
            WAITING INT
        );""")


def log_states(cur_j: Cursor, rsu: str, l: str, chain: List) -> None:
    """ TODO: Finish documentation.

    :param cur_j:
    :param rsu:
    :param l:
    :param chain:
    :return:
    """
    from datetime import datetime

    for state in [x for x in chain if x[0] is not None]:
        cur_j.execute("""
            INSERT INTO WAIT_POP
            VALUES ({});
        """.format(','.join('?' for _ in range(14))),
                      (datetime.now(), rsu, l, state[0].i_0, state[0].big_n, state[0].mu, state[0].s, state[0].kappa,
                       state[0].omega, state[0].u, state[0].v, state[0].m, state[0].p, state[1]))


def metro_hast(it: int, rfs: List, r: int, n: int, m_p: ModelParameters, m_p_sigma: ModelParameters) -> List:
    """ TODO: Finish documentation.

    :param it:
    :param rfs:
    :param r:
    :param n:
    :param m_p:
    :param m_p_sigma:
    :return:
    """
    from numpy.random import uniform, normal
    from numpy import average
    from single import Single
    from compare import compare

    # Create our chain of states and waiting times.
    states, tau, t = [[None, 0] for _ in range(it)], 0, 1

    for j in range(it):
        # Generate and evolve a single population given the current parameters. Compute the delta.
        acceptance_prob, states[tau] = 1 - average(compare(r, n, rfs, Single(m_p).evolve())), [m_p, t]

        if uniform(0, 1) < acceptance_prob:  # Accept our proposal. Stay here and increment the waiting times.
            t, tau = t + 1, tau

        else:  # Reject our proposal. Generate a new parameter set.
            t, tau = 1, tau + 1
            m_p = ModelParameters(i_0=round(normal(m_p.i_0, m_p_sigma.i_0)),
                                  big_n=round(normal(m_p.big_n, m_p_sigma.big_n)),
                                  mu=normal(m_p.mu, m_p_sigma.mu), s=normal(m_p.s, m_p_sigma.s),
                                  kappa=round(normal(m_p.kappa, m_p_sigma.kappa)),
                                  omega=round(normal(m_p.omega, m_p_sigma.omega)),
                                  u=normal(m_p.u, m_p_sigma.u), v=normal(m_p.v, m_p_sigma.v),
                                  m=normal(m_p.m, m_p_sigma.m), p=normal(m_p.p, m_p_sigma.p))

    return states


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect

    parser = ArgumentParser(description='Evolve allele populations with different parameter sets using a grid search.')
    parser.add_argument('-rdb', help='Location of the real database file.', type=str, default='data/real.db')
    parser.add_argument('-edb', help='Location of the database to record to.', type=str, default='data/emcee.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)
    
    paa('-r', 'Number of populations to use to obtain delta.', int)
    paa('-it', 'Number of iterations to run MCMC for.', int)
    paa('-rsu', 'ID of the real sample data set to compare to.', str)
    paa('-l', 'Locus of the real sample to compare to.', str)

    paa('-i_0', 'Starting repeat length of starting ancestor.', int)
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

    n2_m = int(cur_r.execute(""" -- Retrieve the sample size, the number of alleles. --
        SELECT SAMPLE_SIZE
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (args.rsu, args.l, )).fetchone()[0])

    # Perform the MCMC, and record our chain.
    log_states(cur_e, args.rsu, args.l, metro_hast(args.it, freq_r, args.r, n2_m,
                                                   ModelParameters(i_0=args.i_0, big_n=args.big_n, mu=args.mu,
                                                                   s=args.s, kappa=args.kappa, omega=args.omega,
                                                                   u=args.u, v=args.v, m=args.m, p=args.p),
                                                   ModelParameters(i_0=args.i_0_sigma, big_n=args.big_n_sigma,
                                                                   mu=args.mu_sigma, s=args.s_sigma,
                                                                   kappa=args.kappa_sigma, omega=args.omega_sigma,
                                                                   u=args.u_sigma, v=args.v_sigma, m=args.m_sigma,
                                                                   p=args.p_sigma)))
    conn_e.commit()
