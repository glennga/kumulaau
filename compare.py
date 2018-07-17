#!/usr/bin/env python3
from sqlite3 import Cursor
from typing import List, Iterable
from numpy import ndarray


def create_table(cur_j: Cursor) -> None:
    """ Create the table to log the results of our comparisons to.

    :param cur_j: Cursor to the database file to log to.
    :return: None.
    """
    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS DELTA_POP (
            TIME_R TIMESTAMP,
            SIM_SAMPLE_ID TEXT,
            SIM_EFF_ID TEXT,
            REAL_SAMPLE_UID TEXT,
            REAL_LOCUS TEXT,
            DELTA FLOAT
        );""")


def log_deltas(cur_j: Cursor, deltas: ndarray, sei: str, rsu: str, l: str) -> None:
    """ Given the computed differences between two sample's distributions, record each with a
    unique ID into the database.

    :param cur_j: Cursor to the database to log to.
    :param deltas: Computed differences from the sampling.
    :param sei: ID of the simulated population to sample from.
    :param rsu: ID of the real sample data set to compare to.
    :param l: Locus of the real sample to compare to.
    :return: None.
    """
    from string import ascii_uppercase, digits
    from datetime import datetime
    from random import choice

    for delta in deltas:
        # Generate our unique sampling ID, a 20 character string.
        ssi = ''.join(choice(ascii_uppercase + digits) for _ in range(20))

        # Record our results.
        cur_j.execute("""
            INSERT INTO DELTA_POP
            VALUES (?, ?, ?, ?, ?, ?);
        """, (datetime.now(), ssi, sei, rsu, l, delta))


def population_from_count(ruc: Iterable) -> ndarray:
    """ Transform a list of repeat units and counts into a population of repeat units.

    :param ruc: List of lists, whose first element represents the repeat unit and second represents the count.
    :return: Population of repeat units.
    """
    ru = []
    for repeat_unit, c in ruc:
        ru = ru + [repeat_unit] * c
    return ru


def compare(r: int, n2: int, rfs: List, srue: Iterable) -> ndarray:
    """ Given individuals from the effective simulated population and the frequencies of individuals from a real sample,
    sample the same amount from the simulated population 'r' times and determine the differences in distribution for
    each different simulated sample. Involves a butt-ton of transformation and I'm sure that this can be optimized
    below- but given that this is already pretty fast there is no need to do so.

    :param r: Number of times to sample the simulated population.
    :param n2: Sample size of the alleles. Must match the real sample given here.
    :param rfs: Real frequency sample. First column is the length, second is the frequency.
    :param srue: Simulated population of repeat units (one dimensional list).
    :return: None.
    """
    from collections import Counter
    from numpy.random import choice
    from numpy import array, zeros, empty

    # Transform our sampled real population into a dictionary of ints and floats.
    rfs = {int(a[0]): float(a[1]) for a in rfs}

    delta_rs = empty(r)
    for delta_j in range(r):
        # Randomly sample n individuals from population.
        scs = array([choice(srue) for _ in range(n2)])

        # Determine if the maximum length exists in the simulated data set or the real data set.
        omega = max(scs) + 1 if max(scs) > max(rfs.keys()) else max(rfs.keys()) + 1

        # Fit the simulated population into a sparse vector of frequencies.
        scs_counter, sfs_v = Counter(scs), zeros(omega)
        for repeat_unit in scs_counter:
            sfs_v[repeat_unit] = scs_counter[repeat_unit] / n2

        # Fit the real frequency array into a sparse vector.
        rfs_v = zeros(omega)
        for repeat_unit in rfs.keys():
            rfs_v[repeat_unit] = rfs[repeat_unit]

        # For all repeat lengths, determine the sum difference in frequencies. Normalize this to [0, 1].
        delta_rs[delta_j] = sum([abs(sfs_v[j] - rfs_v[j]) for j in range(omega)]) / 2.0

    # Return the differences.
    return delta_rs


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect

    parser = ArgumentParser(description='Sample our simulated population and compare this to a real data set.')
    parser.add_argument('-sdb', help='Location of the simulated database file.', type=str, default='data/simulate.db')
    parser.add_argument('-rdb', help='Location of the real database file.', type=str, default='data/real.db')
    parser.add_argument('-ssdb', help='Location of the database to record data to.', type=str, default='data/sample.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    paa('-r', 'Number of times to sample the simulated population.', int)
    paa('-sei', 'ID of the simulated population to sample from.', str)
    paa('-rsu', 'ID of the real sample data set to compare to.', str)
    paa('-l', 'Locus of the real sample to compare to.', str)
    args = parser.parse_args()  # Parse our arguments.

    # Connect to all of our databases, and create our table if it does not already exist.
    conn_s, conn_r, conn_ss = connect(args.sdb), connect(args.rdb), connect(args.ssdb)
    cur_s, cur_r, cur_ss = conn_s.cursor(), conn_r.cursor(), conn_ss.cursor()
    create_table(cur_ss)

    freq_r = cur_r.execute(""" -- Pull the frequency distribution from the real database. --
        SELECT ELL, ELL_FREQ
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (args.rsu, args.l, )).fetchall()

    count_s = cur_s.execute(""" -- Pull the count distribution from the simulated database. --
        SELECT ELL, ELL_COUNT
        FROM EFF_ELL
        WHERE EFF_ID LIKE ?
    """, (args.sei, )).fetchall()

    n2_m = int(cur_r.execute(""" -- Retrieve the sample size, the number of alleles. --
        SELECT SAMPLE_SIZE
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (args.rsu, args.l, )).fetchone()[0])

    # Execute our sampling and record the results to the simulated database.
    log_deltas(cur_ss, compare(args.r, n2_m, freq_r, population_from_count(count_s)), args.sei, args.rsu, args.l)
    conn_ss.commit(), conn_ss.close()
