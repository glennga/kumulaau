#!/usr/bin/env python3
from sqlite3 import Cursor
from typing import Iterable, Tuple, List
from numpy.random import choice
from numpy.linalg import norm
from numpy import ndarray
from numba import jit, prange


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

    # Record our results.
    cur_j.executemany("""
        INSERT INTO DELTA_POP
        VALUES (?, ?, ?, ?, ?, ?)
    """, ((datetime.now(), ''.join(choice(ascii_uppercase + digits) for _ in range(20)), sei, rsu, l, a)
          for a in deltas))


def population_from_count(ruc: Iterable) -> ndarray:
    """ Transform a list of repeat units and counts into a population of repeat units.

    :param ruc: List of lists, whose first element represents the repeat unit and second represents the count.
    :return: Population of repeat units.
    """
    ru = []
    for repeat_unit, c in ruc:
        ru = ru + [repeat_unit] * c
    return ru


def prepare_delta(rfs_d: List, srue: ndarray, two_n: int, r: int) -> Tuple:
    """ Generate the storage vectors for the sample simulations (both individual and frequency), the storage vector
    for the generated deltas, and sparse frequency vector for the real sample.

    :param rfs_d: Dirty real frequency sample. Needs to be transformed into a sparse frequency vector.
    :param srue: Simulated population of repeat units (one dimensional list).
    :param two_n: Sample size of the alleles.
    :param r: Number of times to sample the simulated population.
    :return: Storage and data vectors in order of: 'scs', 'sfs', 'rfs', 'delta_rs'.
    """
    from numpy import zeros

    rfs_dict = {int(a[0]): float(a[1]) for a in rfs_d}  # Cast our real frequencies into numbers.

    # Determine the omega from the simulated effective population and our real sample.
    omega = max(srue) + 1 if max(srue) > max(rfs_dict.keys()) else max(rfs_dict.keys()) + 1

    # Create the vectors to return.
    scs_out, sfs_out, rfs_out, delta_rs_out = [zeros(two_n), zeros(omega), zeros(two_n), zeros(r)]

    # Fit our real distribution into a sparse frequency vector.
    for repeat_unit in rfs_dict.keys():
        rfs_out[repeat_unit] = rfs_dict[repeat_unit]

    return scs_out, sfs_out, rfs_out, delta_rs_out


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def cosine_delta(scs: ndarray, sfs: ndarray, rfs: ndarray, srue: ndarray, delta_rs: ndarray) -> None:
    """ TODO: Finish

    :param scs:
    :param sfs:
    :param rfs:
    :param srue:
    :param delta_rs:
    :return:
    """
    for delta_k in prange(delta_rs.size):
        for k in prange(scs.size):  # Randomly sample n individuals from population.
            scs[k] = choice(srue)

        # Fit the simulated population into a sparse vector of frequencies.
        for repeat_unit in prange(sfs.size):
            i_count = 0
            for i in scs:  # Ugly code, but I'm trying to avoid memory allocation. ):
                i_count += 1 if i == repeat_unit else 0

            sfs[repeat_unit] = i_count / scs.size

        # Compute the cosine similarity (A dot B / |A||B|), and store this in our output vector.
        delta_rs[delta_k] = sfs.dot(rfs) / (norm(sfs) * norm(rfs))


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def frequency_delta(scs: ndarray, sfs: ndarray, rfs: ndarray, srue: ndarray, delta_rs: ndarray) -> None:
    """ Given individuals from the effective simulated population and the frequencies of individuals from a real sample,
    sample the same amount from the simulated population 'r' times and determine the differences in distribution for
    each different simulated sample. All vectors passed MUST be of appropriate size and must be zeroed out before use.
    Optimized by Numba.

    :param scs: Storage vector, used to hold the sampled simulated population.
    :param sfs: Storage sparse vector, used to hold the frequency sample.
    :param rfs: Real frequency sample, represented as a sparse frequency vector indexed by repeat length.
    :param srue: Simulated population of repeat units (one dimensional list).
    :param delta_rs: Output vector, used to store the computed deltas of each sample.
    :return: None.
    """
    for delta_k in prange(delta_rs.size):
        for k in prange(scs.size):  # Randomly sample n individuals from population.
            scs[k] = choice(srue)

        # Fit the simulated population into a sparse vector of frequencies.
        for repeat_unit in prange(sfs.size):
            i_count = 0
            for i in scs:  # Ugly code, but I'm trying to avoid memory allocation. ):
                i_count += 1 if i == repeat_unit else 0

            sfs[repeat_unit] = i_count / scs.size

        # For all repeat lengths, determine the sum difference in frequencies. Normalize this to [0, 1].
        delta_rs[delta_k] = 0
        for j in prange(sfs.size):
            delta_rs[delta_k] += abs(sfs[j] - rfs[j])
        delta_rs[delta_k] /= 2.0


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect

    parser = ArgumentParser(description='Sample our simulated population and compare this to a real data set.')
    parser.add_argument('-sdb', help='Location of the simulated database file.', type=str, default='data/simulate.db')
    parser.add_argument('-rdb', help='Location of the real database file.', type=str, default='data/real.db')
    parser.add_argument('-ssdb', help='Location of the database to record data to.', type=str, default='data/sample.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    parser.add_argument('-f', help='Similarity index to use to compare.', type=str, choices=['COSINE', 'FREQ'])
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

    two_nm = int(cur_r.execute(""" -- Retrieve the sample size, the number of alleles. --
        SELECT SAMPLE_SIZE
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (args.rsu, args.l, )).fetchone()[0])

    v_0 = population_from_count(count_s)  # Execute the sampling.
    v_1, v_2, v_3, v_4 = prepare_delta(freq_r, v_0, two_nm, args.r)
    frequency_delta(v_1, v_2, v_3, v_4) if args.f.casefold() == 'freq' else cosine_delta(v_1, v_2, v_3, v_4)

    log_deltas(cur_ss, v_4, args.sei, args.rsu, args.l)  # Record to the simulated database.
    conn_ss.commit(), conn_ss.close(), conn_s.close(), conn_r.close()
