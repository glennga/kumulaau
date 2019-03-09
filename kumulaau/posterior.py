#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from sqlite3 import Cursor, Connection
from numpy import ndarray
from typing import List


class MCMCA(ABC):
    @property
    @abstractmethod
    def MODEL_NAME(self):
        """ Enforce the definition of a model name.

        :return: None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def MODEL_SCHEME_SQL(self):
        """ Enforce the definition of some SQL schema for the {MODEL_NAME}_MODEL table.

        :return: None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def PARAMETER_CLASS(self):
        """ Enforce the definition of some parameter class, a child of the Parameter class.

        :return: None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def DISTANCE_CLASS(self):
        """ Enforce the definition of a distance class, a child of the Distance class.

        :return: None.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _sample(theta, i_0) -> ndarray:
        """ Given some parameter set theta and an initial state i_0, return a population that represents sampling from
        the posterior distribution.

        :param theta: Current parameter set to sample.
        :param i_0: Common ancestor state.
        :return: Array of repeat lengths to compare with an observed population.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _walk(theta, pi_epsilon):
        """ Given some parameter set theta and distribution parameters pi_epsilon, generate a new parameter set.

        :param theta: Current point to walk from.
        :param pi_epsilon: Distribution parameters to walk with.
        :return: A new parameter set.
        """
        raise NotImplementedError

    def pull_frequencies(self, cursor_o):
        """ TODO

        :param cursor_o:
        :param uid:
        :param locus:
        :return:
        """
        return list(map(lambda a, b: cursor_o.execute("""
            SELECT ELL, ELL_FREQ
            FROM OBSERVED_ELL
            WHERE SAMPLE_UID LIKE ?
            AND LOCUS LIKE ?
        """, (a, b,)).fetchall(), self.uid, self.locus))

    @classmethod
    def retrieve_last(cls):
        """ Retrieve the last parameter set from a previous run. This is meant to be used for continuing MCMC runs.

        :return: Parameters object holding the parameter set from last recorded run.
        """
        return cls.PARAMETER_CLASS(*cls.cursor.execute(f"""
            SELECT {cls.MODEL_SCHEME_SQL.replace("INT", "").replace("FLOAT", "")}
            FROM {cls.MODEL_NAME}_MODEL
            INNER JOIN {cls.MODEL_NAME}_RESULTS USING (TIME_R)
            ORDER BY {cls.MODEL_NAME}_MODEL.TIME_R, {cls.MODEL_NAME}_MODEL.PROPOSED_TIME DESC
            LIMIT 1
        """).fetchone())

    def create_tables(self) -> None:
        """ Create the tables to log the results of our MCMC to.

        :param cursor: Cursor to the database file to log to.
        :return: None.
        """
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.MODEL_NAME}_OBSERVED (
                TIME_R TIMESTAMP,
                UID_OBSERVED TEXT,
                LOCUS_OBSERVED TEXT
            );""")

        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.MODEL_NAME}_MODEL (
                TIME_R TIMESTAMP,
                {self.MODEL_SCHEME_SQL}
        );""")

        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.MODEL_NAME}_RESULTS (
                TIME_R TIMESTAMP,
                WAITING_TIME INT,
                LIKELIHOOD FLOAT,
                DISTANCE FLOAT,
                PROPOSED_TIME INT
            );""")

    def determine_boundaries(self, seed, iterations_n):
        start = 0 if seed == 0 else self.cursor.execute(f"""
            SELECT PROPOSED_TIME  -- Determine our iteration boundaries. --
            FROM {self.MODEL_NAME}_RESULTS
            ORDER BY PROPOSED_TIME DESC
            LIMIT 1
        """).fetchone()[0]

        return [start, iterations_n + start + 1]

    def __init__(self, connection_m: Connection, connection_o: Connection, theta_0, pi_epsilon, uid_observed: List,
                 locus_observed: List, simulation_n: int, iterations_n: int, flush_n: int, seed: int, epsilon: float):
        """

        :param connection_m: Connection to the database to log to.
        :param connection_o: Connection to the database holding the observed frequencies to compare to.
        :param theta_0: An initial state for the MCMC.
        :param pi_epsilon: Distribution parameters associated with the walk function.
        :param uid_observed: IDs of observed samples to compare to.
        :param locus_observed: Loci of observed samples (must match with uid).
        :param simulation_n: Number of simulations to use to obtain a distance.
        :param iterations_n: Number of iterations to run MCMC for.
        :param flush_n: Number of iterations to run MCMC before flushing to disk.
        :param seed: 1 -> last recorded "mdb" position is used (TIME_R, PROPOSED_TIME).
        :param epsilon: Maximum acceptance value for distance between [0, 1].
        """
        # Save the connection to our logging database.
        self.connection, self.cursor = connection_m, connection_m.cursor()

        # Save the parameters associated with our ABC-MCMC itself.
        self.simulation_n, self.iterations_n, self.epsilon = simulation_n, iterations_n, epsilon

        # Create our tables to log to, and save our logging parameter.
        self.create_tables()
        self.flush_n = flush_n

        # Save our observation database parameters and pull the associated frequencies.
        self.uid, self.locus = uid_observed, locus_observed
        self.observed = self.pull_frequencies(connection_o.cursor())

        # Determine our starting point, and save the walk distribution parameters.
        self.theta_0 = theta_0 if seed != 1 else self.retrieve_last()
        self.pi_epsilon = pi_epsilon

        # Determine the iterations the MCMC will run for.
        self.boundaries = self.determine_boundaries(seed, iterations_n)

    def log_states(self, x: List, i: int) -> None:
        """ Record our states to some database.

        :param x: States and associated times & probabilities collected after running MCMC (our Markov chain).
        :param i: Current iteration of the MCMC run.
        :return: None.
        """
        from datetime import datetime
        date_string = datetime.now()

        if len(x) == 0 and i % self.flush_n != 0:  # Record every flush_n iterations.
            return

        # Record our observed sample log strings and datetime.
        self.cursor.executemany(f"""
            INSERT INTO {self.MODEL_NAME}_OBSERVED
            VALUES (?, ?, ?)
        """, ((date_string, a[0], a[1]) for a in zip(self.uid, self.locus)))

        # noinspection SqlInsertValues
        self.cursor.executemany(f"""
            INSERT INTO {self.MODEL_NAME}_MODEL
            VALUES ({','.join('?' for _ in range(x[0][0].PARAMETER_COUNT + 1))});
        """, ((date_string,) + tuple(a[0]) for a in x))

        # Record the results associated with each model.
        self.cursor.executemany(f"""
            INSERT INTO {self.MODEL_NAME}_RESULTS
            VALUES (?, ?, ?, ?, ?)
        """, ((date_string, a[1], a[2], a[3], a[4]) for a in x))
        self.connection.commit()

        # Clear our chain except for the last state.
        x[:] = [x[-1]]

    def cleanup(self) -> None:
        """ Remove the initial state logged from our database, required to start the MCMC.

        :return: None.
        """
        self.cursor.execute(f"""
            DELETE FROM {self.MODEL_NAME}_MODEL
            WHERE TIME_R IN (
                SELECT TIME_R 
                FROM {self.MODEL_NAME}_RESULTS
                WHERE PROPOSED_TIME = 0
        );""")
        self.cursor.execute(f"""
            DELETE FROM {self.MODEL_NAME}_RESULTS
            WHERE PROPOSED_TIME = 0;
        """)

    def run(self) -> None:
        """ A MCMC algorithm to approximate the posterior distribution of a generic model, whose acceptance to the
        chain is determined by some distance between repeat length distributions. My interpretation of this
        ABC-MCMC approach is given below:

        1) We start with some initial guess theta_0. Right off the bat, we move to another theta from theta_0.
        2) For 'iterations_bounds[1] - iterations_bounds[0]' iterations...
            a) For 'simulation_n' iterations...
                i) We simulate a population using the given theta.
                ii) For each observed frequency ... 'D'
                    1) We compute the difference between the two distributions.
                    2) If this difference is less than our epsilon term, we add 1 to a vector modeling D.
            b) Compute the probability that each observed matches a generated population: all of D / simulation_n.
            c) If this probability is greater than the probability of the previous, we accept.
            d) Otherwise, we accept our proposed with probability p(proposed) / p(prev).

        :return: None.
        """
        from numpy.random import uniform

        # Seed our Markov chain with our initial guess.
        x = [[self.theta_0, 1, 1.0e-10, 1.0e-10, 0]]

        for i in range(self.boundaries[0] + 1, self.boundaries[1]):
            # Walk from our previous state.
            theta_proposed = self.PARAMETER_CLASS.from_walk(x[-1][0], self.pi_epsilon, self._walk)

            # Populate our delta matrix.
            delta = self.DISTANCE_CLASS(self.observed, theta_proposed.kappa, theta_proposed.omega, self.simulation_n)

            # Compute our matched and delta matrix (simulation (rows) by observation (columns)). Get a mean distance.
            expected_distance = delta.fill_matrices(self._sample, theta_proposed, self.epsilon)

            # Accept our proposal according to our alpha value.
            p_proposed, p_k = delta.match_likelihood(), x[-1][2]
            if p_proposed / p_k > uniform(0, 1):
                x = x + [[theta_proposed, 1, p_proposed, expected_distance, i]]

            # Reject our proposal. We keep our current state and increment our waiting times.
            else:
                x[-1][1] += 1

            # We record to our chain. This is dependent on the current iteration of MCMC.
            self.log_states(x, i)

        # Remove our initial guess.
        self.cleanup()

