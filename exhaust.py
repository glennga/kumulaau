#!/usr/bin/env python3
from sqlite3 import Cursor
from single import Single, ModelParameters


def create_table(cur_j: Cursor) -> None:
    """ Create the table to log our simulated populations to.

    :param cur_j: Cursor to the database file to log to.
    :return: None.
    """
    cur_j.execute("""  -- Holds the effective population data associated with each allele. --
        CREATE TABLE IF NOT EXISTS EFF_ELL (
            EFF_ID TEXT,
            ELL INT,
            ELL_COUNT INT,
            ELL_FREQ FLOAT
        );""")

    cur_j.execute(""" -- Holds the effective population data associated with each population itself. --
        CREATE TABLE IF NOT EXISTS EFF_POP (
            EFF_ID TEXT,
            TIME_R TIMESTAMP,
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
            MEAN_ELL FLOAT,
            STD_ELL FLOAT
    );""")


def log_eff(cur_j: Cursor, z_j: Single) -> None:
    """ Record our effective population to the database.

    :param cur_j: Cursor to the database file to log to.
    :param z_j: Population object holding our ancestor list.
    :return: None.
    """
    from string import ascii_uppercase, digits
    from numpy import average, std
    from collections import Counter
    from datetime import datetime
    from random import choice

    # Generate our unique simulation ID, a 20 character string.
    eff_id = ''.join(choice(ascii_uppercase + digits) for _ in range(20))

    # Group the allele variations together by microsatellite repeat length, and compute the count and frequency of each.
    ell_counter = Counter(z_j.ell_evolved)
    for i in set(ell_counter):
        cur_j.execute("""
            INSERT INTO EFF_ELL
            VALUES (?, ?, ?, ?);
        """, (eff_id, int(i), ell_counter[i], ell_counter[i] / len(z_j.ell_evolved)))

    # Record the parameters associated with the effective population, and some basic stats (average, std, n_mu).
    cur_j.execute("""
        INSERT INTO EFF_POP
        VALUES ({});
    """.format(','.join('?' for _ in range(13))), (eff_id, datetime.now(), z_j.i_0, z_j.big_n, z_j.mu, z_j.s, z_j.kappa,
                                                   z_j.omega, z_j.u, z_j.v, z_j.m, z_j.p, average(z_j.ell_evolved),
                                                   std(z_j.ell_evolved)))


if __name__ == '__main__':
    from argparse import ArgumentParser
    from itertools import product
    from sqlite3 import connect

    parser = ArgumentParser(description='Evolve allele populations with different parameter sets using a grid search.')    
    parser.add_argument('-db', help='Location of the database file.', type=str, default='data/simulate.db')
    parser.add_argument('-r', help='Number of populations to generate given the same parameter.', type=int, default=1)
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3, nargs='+')

    paa('-i_0', 'Repeat lengths of starting ancestor.', int)
    paa('-big_n', 'Effective population sizes.', int)
    paa('-mu', 'Mutation rates, bounded by (0, infinity).', float)
    paa('-s', 'Proportional rates, bounded by (-1 / (omega - kappa + 1), infinity).', float)
    paa('-kappa', 'Lower bounds of possible repeat lengths.', int)
    paa('-omega', 'Upper bounds of possible repeat lengths.', int)
    paa('-u', 'Constant bias parameters, bounded by [0, 1].', float)
    paa('-v', 'Linear bias parameters, bounded by (-infinity, infinity).', float)
    paa('-m', 'Success probabilities for truncated geometric distribution.', float)
    paa('-p', 'Probabilities that the repeat length change is +/- 1.', float)
    args = parser.parse_args()  # Parse our arguments.

    # Connect to our database, and create our table if it does not already exist.
    conn = connect(args.db)
    cur = conn.cursor()
    create_table(cur)

    # Compute the cartesian product of all argument sets. This devolves to a 10-dimensional grid search.
    for a in list(product(args.i_0, args.big_n, args.mu, args.s, args.kappa, args.omega, args.u,
                          args.v, args.m, args.p)):
        for _ in range(args.r):

            # Evolve each population 'r' times with the same parameter.
            z = Single(ModelParameters(i_0=a[0], big_n=a[1], mu=a[2], s=a[3], kappa=a[4], omega=a[5],
                                       u=a[6], v=a[7], m=a[8], p=a[9]))
            z.evolve()

            # Record this to our database.
            log_eff(cur, z)

    conn.commit()  # Record our runs and exit.
    print('Grid search done!')
