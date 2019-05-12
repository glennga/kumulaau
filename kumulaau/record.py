#!/usr/bin/env python3
from typing import Sequence, Iterable, List


class RecordSQLite(object):
    # Schema for the observed table. This is fixed.
    _OBSERVED_SCHEMA = 'RUN_R TEXT, ' \
                       'POP_ID TEXT, ' \
                       'ELL INT, ' \
                       'ELL_FREQ FLOAT '

    # Schema for the results table. This is fixed.
    _RESULTS_SCHEMA = 'RUN_R TEXT, ' \
                      'TIME_R TIMESTAMP, ' \
                      'WAITING_TIME INT, ' \
                      'P_PROPOSED FLOAT, ' \
                      'EXPECTED_DELTA FLOAT, ' \
                      'PROPOSED_TIME INT '

    # Schema for the experiment table. This is fixed. We are just casting everything to text.
    _EXPR_SCHEMA = 'RUN_R TEXT, ' \
                   'FIELD_NAME TEXT, ' \
                   'FIELD_VAL TEXT '

    def __init__(self, filename: str, model_name: str, model_schema: str, is_new_run: bool):
        """ There exists four tables here: the observed table, the model table, the results table, and the run table.

        (a) The first will always maintain the same schema of UID, LOCUS pairing.
        (b) The second is dependent on the model itself, holding all parameters associated with the model.
        (c) The third is dependent on the Markov chain, holding all results produced.
        (d) The fourth is dependent on any experiment parameters that is to be added.

        All tables have a primary key of RUN_R, a randomly generated 10 character key. The model and results table are
        keyed compositely: (RUN_R, TIME_R).

        :param filename: Location of the results database to record to.
        :param model_name: Prefix to append to all tables associated with this model.
        :param model_schema: Schema of the _MODEL table.
        :param is_new_run: Flag which indicates if the current run to be logged is new or not.
        :return: None.
        """
        from sqlite3 import connect

        # Connect to our database.
        self.connection = connect(filename)
        self.cursor = self.connection.cursor()

        # Determine our table names.
        self.observed_table = model_name + '_OBSERVED'
        self.model_table = model_name + '_MODEL'
        self.results_table = model_name + '_RESULTS'
        self.expr_table = model_name + '_EXPR'

        # Create the tables if they do not already exist.
        list(map(lambda a, b: self._create_table(a, b),
                 [self.observed_table, self.model_table, self.results_table, self.expr_table],
                 [self._OBSERVED_SCHEMA, 'RUN_R TEXT, TIME_R TIMESTAMP, ' + model_schema, self._RESULTS_SCHEMA,
                  self._EXPR_SCHEMA]))

        # Determine our fields.
        self.model_fields = self._parse_fields(self.model_table)
        self.results_fields = self._parse_fields(self.results_table)
        self.expr_fields = self._parse_fields(self.expr_table)
        self.model_name = model_name

        # Determine the run key.
        self.run_r = self._generate_run_key() if is_new_run else self.retrieve_last_result('RUN_R')

    def __enter__(self):
        """ Required to use class as context manager.

        :return: Self.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Required to use class as context manager. Remove any invalid records and close our connection to our
        database.

        :param exc_type: Exception type (not used).
        :param exc_val: Exception value (not used).
        :param exc_tb: Exception throwback (not used).
        :return: None.
        """
        self.cursor.execute(f"""
            DELETE FROM {self.model_table}
            WHERE TIME_R IN (
                SELECT TIME_R
                FROM {self.results_table}
                WHERE TIME_R = 0
            );
        """)
        self.cursor.execute(f"""
            DELETE FROM {self.results_table}
            WHERE TIME_R = 0;
        """)

        # Commit and close our database connection.
        self.connection.commit(), self.connection.close()

    def record_observed(self, observations: Sequence, pop_ids: Iterable = None):
        """ Given a set of observations in tuple form, record this to the _OBSERVED table. If pop_ids are specified,
        use these for the POP_ID fields. Otherwise, enumerate our given populations starting from 1.

        :param observations: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
        :param pop_ids: Optional POP_IDs list to attach to each observed population.
        :return: None.
        """
        if pop_ids is None:  # If not specified, create our POP_IDs to uniquely identify each field.
            pop_ids = [str(a) for a in range(1, len(observations) + 1)]

        for population, pop_id in zip(observations, pop_ids):
            self.cursor.executemany(f"""
                INSERT INTO {self.observed_table}
                VALUES (?, ?, ?, ?);
            """, ((self.run_r, pop_id, a[0], a[1]) for a in population))

    def record_expr(self, field_names: Sequence, field_vals: Sequence):
        """ Given a set of field names and corresponding values, record these to the _EXPR table. We cast everything to
        text here.

        :param field_names: Field names to record, whose order matches with field_vals.
        :param field_vals: Field values to record, whose order matches with field_names.
        :return: None.
        """
        self.cursor.executemany(f"""
            INSERT INTO {self.expr_table}
            VALUES (?, ?, ?)
        """, zip([self.run_r for _ in field_names], [str(a) for a in field_names], [str(a) for a in field_vals]))

    def handler_factory(self, flush_n: int):
        """ Handler factory for a sequence of records, of arbitrary type. We specify how often we flush our record
        set to dish.

        :param flush_n: Number of iterations to run before flushing to disk.
        :return: None.
        """
        def _handler(x: Sequence, i: int):
            if i % flush_n != 0 or len(x) == 0:  # Record every flush_n iterations.
                return

            # Record to our _MODEL table. Do not log our initial result.
            self.cursor.executemany(f"""
                INSERT INTO {self.model_table}
                VALUES ({','.join('?' for _ in self.model_fields)});
            """, ((self.run_r, a.time_r) + tuple([getattr(a.theta, b) for b in self.model_fields[2:]]) for a in x[1:]))

            # Record to our _RESULTS table. Do not log our initial result.
            self.cursor.executemany(f"""
                INSERT INTO {self.results_table}
                VALUES ({','.join('?' for _ in self.results_fields)});
            """, ((self.run_r,) + tuple([getattr(a, b) for b in self.results_fields[1:]]) for a in x[1:]))

            # Remove every record except the last.
            x[:] = [x[-1]]

            # Record our changes on disk.
            self.connection.commit()

        return _handler

    def retrieve_last_theta(self):
        """ Query our _MODEL table for the last recorded parameter set according to TIME_R.

        :return: A dictionary consisting of the last recorded parameter set.
        """
        return dict(zip(self.model_fields[2:], self.cursor.execute(f"""
            SELECT {','.join(a.upper() for a in self.model_fields[2:])}
            FROM {self.model_table}
            ORDER BY TIME_R DESC
            LIMIT 1
        """).fetchone()))

    def retrieve_last_result(self, select_clause: str):
        """ Given a select clause, query our _RESULTS table for the last recorded result according to TIME_R.

        :param select_clause: Item to query and return.
        :return: Tuple of result or the sole item itself if the select clause only specifies one field.
        """
        result = self.cursor.execute(f"""
            SELECT {select_clause}
            FROM {self.results_table}
            ORDER BY TIME_R DESC
            LIMIT 1
        """).fetchone()

        if result is None:  # Ensure that we have results to start with.
            raise RuntimeError("Unable to retrieve last result. Is the database seeded?")

        return result[0] if len(result) == 1 else result

    def _create_table(self, name: str, schema: str) -> None:
        """ Given the name of the table to create and the specific schema, run a DDL and do  not throw errors if it
        already exists.

        :param name: Name of the table to create.
        :param schema: Schema of the table to create.
        :return: None.
        """
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
            {schema}
        );""")

    def _parse_fields(self, name: str) -> List:
        """ Given the name of the table in an existing database, retrieve the field names.

        :param name: Name of the table to extract the fields from.
        :return: List of fields associated with the given table.
        """
        return list(map(lambda a: a[1].lower(), self.cursor.execute(f"""
            PRAGMA table_info({name});
        """).fetchall()))

    @staticmethod
    def _generate_run_key() -> str:
        """ Generate a random 10 digit alphanumeric code to associate with some run to record.

        :return: Random 10 digit alphanumberic string.
        """
        from string import ascii_uppercase, digits
        from random import choice

        return ''.join(choice(ascii_uppercase + digits) for _ in range(10))
