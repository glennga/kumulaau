#!/usr/bin/env python3
from typing import List, Iterable, Callable
from numpy import ndarray, array
from argparse import Namespace
from sqlite3 import Cursor

# The name of the tables in the alfred and record databases.
_ALFRED_TABLE_NAME = 'OBSERVED_ELL'

# Our table schema for the alfred database.
_ALFRED_SCHEMA = 'TIME_R TIMESTAMP, POP_NAME TEXT, POP_UID TEXT, SAMPLE_UID TEXT, SAMPLE_SIZE INT, \
          LOCUS TEXT, ELL INT, ELL_FREQ FLOAT'

# Our table fields for the alfred database.
ALFRED_FIELDS = 'TIME_R, POP_NAME, POP_UID, SAMPLE_UID, SAMPLE_SIZE, LOCUS, ELL, ELL_FREQ'


def _extract_tuples(cursor: Cursor, uid_loci: Iterable) -> List:
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


def extract_alfred_tuples(uid_loci: Iterable, filename: str = 'data/observed.db') -> List:
    """ Wrapper for the _extract_tuples method. Here we verify that the ALFRED script has been run, connect to
    our observation database, and return the results from _extract_tuples.

    :param uid_loci: List of (uid, loci) pairs to query our database with. Order of tuple matters here!!
    :param filename: Location of the observation database.
    :return: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    """
    from sqlite3 import connect

    connection = connect(filename)
    cursor = connection.cursor()

    # Verify that the ALFRED script has been run.
    if not bool(cursor.execute(f"""
        SELECT NAME
        FROM sqlite_master
        WHERE type='table' AND NAME='{_ALFRED_TABLE_NAME}'
    """).fetchone()):
        raise LookupError(f"'{_ALFRED_TABLE_NAME}' not found in observation database.")

    # Extract our tuples from our database.
    tuples = _extract_tuples(cursor, uid_loci)

    # Return a 2D list of (length, frequency) tuples.
    connection.close()
    return tuples


def generate_tuples(generator: Callable, observation_n: int) -> List:
    """ Given a function that generates repeat lengths (e.g. a model's sample function), repeat this process
    observation_n times and transform all resulting runs into the tuple representation.

    :param generator: Function that generates a collection of repeat lengths.
    :param observation_n: Number of times to run generator for.
    :return: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    """
    from collections import Counter

    to_tuples = lambda a: [(b[0], b[1] / len(a)) for b in Counter(a).items()]
    return [to_tuples(generator()) for _ in range(observation_n)]


def tuples_to_dictionaries(tuples: Iterable) -> List:
    """ Generate the dictionary representation (frequencies indexed by repeat lengths) using the tuple
    representation.

    :param tuples: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :return: 1D list of dictionaries representing the (repeat length: frequency) dictionaries.
    """
    return [{int(a[0]): float(a[1]) for a in b} for b in tuples]


def tuples_to_distribution_vector(tuples: Iterable, sample_n: int) -> ndarray:
    """ Generate the distribution vector representation (row = observation, column = distribution) using the tuple
    representation and user-defined boundaries.

    :param tuples: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :param sample_n: Number of samples to generate per distribution.
    :return: Array of numpy arrays holding a set of repeat lengths.
    """
    from numpy import repeat

    return array([repeat(array(list(observation.keys())), array([int(a * sample_n) for a in observation.values()]))
                  for observation in tuples_to_dictionaries(tuples)])


def tuples_to_pool(tuples: Iterable) -> List:
    """ Using the tuples representation, parse all repeat lengths that exist in this specific observation.

    :param tuples: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :return: A 1D list of all repeat lengths associated with this observation.
    """
    return list(set([a[0] for b in tuples for a in b]))


def create_record_uid_loci_table(cursor: Cursor, table_name: str, pk_name_type: str) -> None:
    """ Given a cursor to some database, the name of the table, and the primary key associated with the table, create
    a table with the schema: {pk}, UID TEXT, LOCI TEXT.

    :param cursor: Cursor to the database to create the table for.
    :param table_name: Name of the table to create.
    :param pk_name_type: Primary key field and type.
    :return: None.
    """
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
        {pk_name_type}, UID TEXT, LOCI TEXT
    );""")


def record_uid_loci(cursor: Cursor, table_name: str, pk, uid_loci: Iterable) -> None:
    """ Given a cursor to some database, the name of the table, and the primary key associated with this insert, insert
    our uid_loci pairs appropriately.

    :param cursor: Cursor to the database containing the table to log to.
    :param table_name: Name of the table to log to. Table must already exist.
    :param pk: Primary key.
    :param uid_loci: List of (uid, loci) pairs to query our database with. Order of tuple matters here!!
    :return: None.
    """
    cursor.executemany(f"""
        INSERT INTO {table_name}
        VALUES (?, ?, ?);
    """, ((pk, a[0], a[1]) for a in uid_loci))


def create_alfred_table(cursor: Cursor) -> None:
    """ Given a cursor to ALFRED database, create the ALFRED table.

    :param cursor: Cursor to the observation database to create the table for.
    :return: None.
    """
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {_ALFRED_TABLE_NAME} (
        {_ALFRED_SCHEMA}
    );""")


def record_to_alfred_table(cursor: Cursor, record) -> None:
    """ Given a cursor to the ALFRED database, record some field.

    :param cursor: Cursor to the observation database to record to.
    :param record: Namespace holding all required fields to record.
    :return: None.
    """
    fields = [b.lower().strip() for b in ALFRED_FIELDS.split(',')]

    cursor.execute(f"""
        INSERT OR REPLACE INTO {_ALFRED_TABLE_NAME}
        VALUES ({','.join('?' for _ in fields)});
    """, tuple(getattr(record, a) for a in fields))


def get_arguments() -> Namespace:
    """ Create the CLI and parse the arguments.

    :return: Namespace of all values.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Observation parser for ALFRED database.')

    list(map(lambda a: parser.add_argument(a[0], help=a[1], type=a[2], nargs=a[3], default=a[4], choices=a[5]), [
        ['-odb', 'Location of the observed database file.', str, None, 'data/observed.db', None],
        ['-uid', 'IDs of observed samples to compare to.', str, '+', None, None],
        ['-loci', 'Loci of observed samples (must match with uid).', str, '+', None, None],
    ]))

    return parser.parse_args()


if __name__ == '__main__':
    arguments = get_arguments()  # Parse our arguments.

    print(extract_alfred_tuples(zip(arguments.uid, arguments.loci), arguments.odb))
