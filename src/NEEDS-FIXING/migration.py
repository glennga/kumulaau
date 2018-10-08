#!/usr/bin/env python3
from abc import ABC, abstractmethod
from sqlite3 import Cursor


def create_table(cur_j: Cursor) -> None:
    """ Create the tables to log the results of some population model MCMC to. Our data model is normalized to
    accommodate different population migration models. Each model varies in which real samples are assigned to each
    population, and what parameters exist in the first place.

    :param cur_j: Cursor to the database file to log to.
    :return: None.
    """
    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_REAL (
            TIME_R TIMESTAMP,
            REAL_SAMPLE_UID TEXT,
            REAL_LOCUS TEXT,
            POP TEXT
        );""")

    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_N (
            TIME_R TIMESTAMP,
            ACCEPTANCE_TIME INT,
            POP TEXT,
            N_TYPE TEXT, 
            N_VALUE INT
        );""")

    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_MIGRATION (
            TIME_R TIMESTAMP,
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


class Migration(ABC):
    @abstractmethod
    def mcmc(self, it: int, rs: int, rp: int, epsilon: float) -> None:
        """ All migration population models must implement a MCMC. What varies is the parameter space, and how they
        compute the distance of the samples and the simulation.

        :param it: Number of iterations to run MCMC for.
        :param rs: Number of samples per simulation to use to obtain delta.
        :param rp: Number of simulations to use to obtain delta.
        :param epsilon: Minimum acceptance value for delta.
        :return: None.
        """
        raise NotImplementedError

    @abstractmethod
    def log_states(self, cur_j: Cursor) -> None:
        """ All migration population models must implement a chain logging method, to record the results of their
        MCMC to the schema defined in the 'create_tables' function above.

        :param cur_j: Cursor to the database file to log to.
        :return: None.
        """
    raise NotImplementedError


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect
    from avna import AVNA
    from json import loads

    parser = ArgumentParser(description='ABC MCMC for the *human migration model* parameter estimation.')
    parser.add_argument('-rdb', help='Location of the real database file.', type=str, default='data/real.db')
    parser.add_argument('-edb', help='Location of the database to record to.', type=str, default='data/migration.db')
    parser.add_argument('-mig', help='Type of migration model to run.', type=str, choices=['AVNA'])
    parser.add_argument('-json', help='Location of the configuration file attached to the migration model.', type=str)
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    paa('-rp', 'Number of simulations to use to obtain delta.', int)
    paa('-rs', 'Number of samples per simulation to use to obtain delta.', int)
    paa('-epsilon', 'Minimum acceptance value for delta.', float)
    paa('-it', 'Number of iterations to run MCMC for.', int)

    c_args = parser.parse_args()  # Parse our arguments.
    j_args = loads(c_args.json)

    # Connect to all of our databases.
    conn_r, conn_e = connect(c_args.rdb), connect(c_args.edb)
    cur_r, cur_e = conn_r.cursor(), conn_e.cursor()
    create_table(cur_e)

    afs_d = cur_r.execute(""" -- Retrieve all known frequency distributions. --
        SELECT ELL, ELL_FREQ
        FROM REAL_ELL
    """).fetchall()

    population_model = None  # Perform the appropriate migration function.
    if c_args.mig.casefold == 'avna':
        population_model = AVNA(j_args, afs_d, cur_e)

    population_model.mcmc(c_args.it, c_args.rs, c_args.rp, c_args.epsilon)  # Run the MCMC, and record to disk.
    population_model.log_states(cur_e)

    conn_e.commit(), conn_r.close(), conn_e.close()
