#!/usr/bin/env python3
from sqlite3 import Cursor
from numpy import ndarray, array
from typing import List

# The name of the observed table.
_TABLE_NAME = 'OBSERVED_ELL'

# Our table schema.
_SCHEMA = 'TIME_R TIMESTAMP, POP_NAME TEXT, POP_UID TEXT, SAMPLE_UID TEXT, SAMPLE_SIZE INT, \
          LOCUS TEXT, ELL TEXT, ELL_FREQ FLOAT'

# Our table fields.
_FIELDS = 'TIME_R, POP_NAME, POP_UID, SAMPLE_UID, SAMPLE_SIZE, LOCUS, ELL, ELL_FREQ'


def _extract_tuples(cursor: Cursor, uid_loci: List) -> List:
    """ Query the observation table for (repeat length, frequency) tuples for various uid, locus pairs.

    :param cursor: Cursor to the observation database.
    :param uid_loci: List of (uid, loci) pairs to query our database with. Order of tuple matters here!!
    :return: 2D List of (int, float) tuples representing the (repeat length, frequency) tuples.
    """
    return list(map(lambda a: cursor.execute("""
        SELECT CAST(ELL AS INTEGER), CAST(ELL_FREQ AS FLOAT)
        FROM OBSERVED_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (a[0], a[1])).fetchall(), uid_loci))


def extract_tuples(filename: str, uid_loci: List) -> List:
    """ Wrapper for the _extract_tuples method. Here we verify that the ALFRED script has been run, connect to
    our observation database, and return the results from _extract_tuples.

    :param filename: Location of the observation database.
    :param uid_loci: List of (uid, loci) pairs to query our database with. Order of tuple matters here!!
    :return: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    """
    from sqlite3 import connect

    connection = connect(filename)
    cursor = connection.cursor()

    # Verify that the ALFRED script has been run.
    if not bool(cursor.execute(f"""
        SELECT NAME
        FROM sqlite_master
        WHERE type='table' AND NAME='{_TABLE_NAME}'
    """).fetchone()):
        raise LookupError("'OBSERVED_ELL' not found in observation database. Run the ALFRED script.")

    # Extract our tuples from our database.
    tuples = _extract_tuples(cursor, uid_loci)

    # Return a 2D list of (length, frequency) tuples.
    connection.close()
    return tuples


def tuples_to_numpy(tuples: List) -> ndarray:
    """ Generate the 2D (int, float) numpy representation using the tuple representation.

    :param tuples: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :return: 2D numpy array of (int, float) tuples representing the (repeat length, frequency) tuples.
    """
    return array([array([(int(a[0]), float(a[1]), ) for a in b]) for b in tuples])


def tuples_to_dictionaries(tuples: List) -> ndarray:
    """ Generate the dictionary representation (frequencies indexed by repeat lengths) using the tuple
    representation.

    :param tuples: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :return: 1D list of dictionaries representing the (repeat length: frequency) dictionaries.
    """
    return array([{int(a[0]): float(a[1]) for a in b} for b in tuples])


def tuples_to_sparse_matrix(tuples: List, bounds: List) -> ndarray:
    """ Generate the sparse matrix representation (column = repeat length, row = observation) using the tuple
    representation and user-defined boundaries.

    :param tuples: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :param bounds: Upper and lower bound (in that order) of the repeat unit space.
    :return: 2D list of frequencies (the sparse frequency matrix).
    """
    from numpy import zeros

    # Generate a dictionary representation.
    observation_dictionary = tuples_to_dictionaries(tuples)

    # Fit our observed distribution into a sparse frequency vector.
    observations = array([zeros(bounds[1] - bounds[0] + 1) for _ in tuples])
    for j, observation in enumerate(observation_dictionary):
        for repeat_unit in observation.keys():
            observations[j, repeat_unit - bounds[0] + 1] = observation[repeat_unit]

    return observations


def tuples_to_pool(tuples: List) -> ndarray:
    """ Using the tuples representation, parse all repeat lengths that exist in this specific observation.

    :param tuples: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :return: A 1D array of all repeat lengths associated with this observation.
    """
    return tuples_to_numpy(tuples)[:, :, 0].flatten()
