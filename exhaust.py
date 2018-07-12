#!/usr/bin/env python3
from sqlite3 import Cursor
from single import Population


def create_table(cur_j: Cursor) -> None:
    """ Create the table to log our simulated to.

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
            KAPPA INT,
            I_0 INT,
            MU_U FLOAT,
            MU_D FLOAT,
            MEAN_ELL FLOAT,
            STD_ELL FLOAT
    );""")


def log_eff(cur_j: Cursor, z_j: Population) -> None:
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
    ell_counter = Counter(z_j.ell_last)
    for i in set(ell_counter):
        cur_j.execute("""
            INSERT INTO EFF_ELL
            VALUES (?, ?, ?, ?);
        """, (eff_id, int(i), ell_counter[i], ell_counter[i] / len(z_j.ell_last)))

    # Record the parameters associated with the effective population, and some basic stats (average, std, n_mu).
    cur_j.execute("""
        INSERT INTO EFF_POP
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (eff_id, datetime.now(), z_j.kappa, z_j.i_0, z_j.mu_u, z_j.mu_d, average(z_j.ell_last), std(z_j.ell_last)))


if __name__ == '__main__':
    from argparse import ArgumentParser
    from itertools import product
    from sqlite3 import connect

    parser = ArgumentParser(description='Evolve allele populations with different parameter sets using a grid search.')
    parser.add_argument('-db', help='Location of the database file.', type=str, default='data/simulated.db')
    parser.add_argument('-r', help='Number of populations to generate given the same parameter.', type=int, default=1)

    parser.add_argument('-kappa', help='Repeat length stationary lower bounds.', type=int, nargs='+', required=True)
    parser.add_argument('-n', help='Effective population sizes.', type=int, nargs='+', required=True)
    parser.add_argument('-i_0', help='Repeat lengths of starting ancestors.', type=int, nargs='+', required=True)
    parser.add_argument('-mu_u', help='Upward mutation rates. Bounded in [0, 1].', type=float, nargs='+', required=True)
    parser.add_argument('-mu_d', help='Downward mutation rates. Bounded in [0, 1].', type=float, nargs='+',
                        required=True)
    args = parser.parse_args()  # Parse our arguments.

    # Connect to our database, and create our table if it does not already exist.
    conn = connect(args.db)
    cur = conn.cursor()
    create_table(cur)

    # Compute the cartesian product of all argument sets. This devolves to a 5-dimensional grid search.
    for p in list(product(args.kappa, args.n, args.i_0, args.mu_u, args.mu_d)):
        for _ in range(args.r):

            # Evolve each population 'r' times with the same parameter.
            z = Population(kappa=p[0], i_0=p[2], mu_u=p[3], mu_d=p[4])
            z.evolve(p[1])

            # Record this to our database.
            log_eff(cur, z)

    conn.commit()  # Record our runs and exit.
    print('Grid search done!')
