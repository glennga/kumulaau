#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from sqlite3 import Connection
from numpy import ndarray
from typing import List


class Posterior(ABC):
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
    def POPULATION_CLASS(self):
        """ Enforce the definition of some population class, a child of the Population class.

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
    def _walk(theta, pi_epsilon):
        """ Given some parameter set theta and distribution parameters pi_epsilon, generate a new parameter set.

        :param theta: Current point to walk from.
        :param pi_epsilon: Distribution parameters to walk with.
        :return: A new parameter set.
        """
        raise NotImplementedError

    def _pull_frequencies(self, cursor_o) -> List:
        """ Given a cursor to the observed database, pull the frequencies associated with the given (uid, locus)
        list.

        :param cursor_o: Cursor to the observed database.
        :return: List of frequencies associated with each (uid, locus) tuple.
        """
        return list(map(lambda a, b: cursor_o.execute("""
            SELECT ELL, ELL_FREQ
            FROM OBSERVED_ELL
            WHERE SAMPLE_UID LIKE ?
            AND LOCUS LIKE ?
        """, (a, b,)).fetchall(), self.uid, self.locus))

    def _sample(self, theta, i_0) -> ndarray:
        """ Given some parameter set theta and an initial state i_0, return a population that represents sampling from
        the posterior distribution.

        :param theta: Current parameter set to sample.
        :param i_0: Common ancestor state.
        :return: Array of repeat lengths to compare with an observed population.
        """
        return self.POPULATION_CLASS(theta).evolve(i_0)

    def __init__(self, connection_m: Connection, connection_o: Connection, uid: List, locus: List):
        """ All posterior classes require the following:

        1. A connection to the logging database.
        2. Frequencies to compare results to.

        :param connection_m: Connection to the database to log to.
        :param connection_o: Connection to the database holding the observed frequencies to compare to.
        :param uid: IDs of observed samples to compare to.
        :param locus: Loci of observed samples (must match with uid).
        """
        # Save the connection to our logging database.
        self.connection, self.cursor = connection_m, connection_m.cursor()

        # Save our observation database parameters and pull the associated frequencies.
        self.uid, self.locus = uid, locus
        self.observed = self._pull_frequencies(connection_o.cursor())


class MCMCA(Posterior, ABC):
    def _create_tables(self) -> None:
        """ Create the tables to log the results of our MCMC to.

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

    def _retrieve_last(self):
        """ Retrieve the last parameter set from a previous run. This is meant to be used for continuing MCMC runs.

        :return: Parameters object holding the parameter set from last recorded run.
        """
        return self.PARAMETER_CLASS(*self.cursor.execute(f"""
            SELECT {self.MODEL_SCHEME_SQL.replace("INT", "").replace("FLOAT", "")}
            FROM {self.MODEL_NAME}_MODEL
            INNER JOIN {self.MODEL_NAME}_RESULTS USING (TIME_R)
            ORDER BY {self.MODEL_NAME}_MODEL.TIME_R, {self.MODEL_NAME}_RESULTS.PROPOSED_TIME DESC
            LIMIT 1
        """).fetchone())

    def _determine_boundaries(self, iterations_n: int, seed: bool) -> List:
        """ Given a seed flag and the number of iterations to run our MCMC for, determine the boundaries of the
        MCMC iterations (start and end).

        :param iterations_n: Number of iterations to run MCMC for.
        :param seed: If raised, pull the last recorded iteration from our database.
        :return: Starting and end iteration counts for our MCMC.
        """
        start = 0 if not seed else self.cursor.execute(f"""
            SELECT PROPOSED_TIME  -- Determine our iteration boundaries. --
            FROM {self.MODEL_NAME}_RESULTS
            ORDER BY PROPOSED_TIME DESC
            LIMIT 1
        """).fetchone()[0]

        return [start, iterations_n + start + 1]

    def __init__(self, connection_m: Connection, connection_o: Connection, uid_observed: List, locus_observed: List,
                 simulation_n: int, iterations_n: int, flush_n: int, epsilon: float, pi_epsilon, theta_0=None):
        """ A lot of shit is done here. I think I may need to break this down deeper in the future, but... I guess for
        now this works:

        1. Pull the frequencies associated with observation parameters we passed in (from Posterior).
        2. Create the tables we must log to.
        3. Determine the starting point, based on the seed parameter (from Posterior).
        4. Determine our iteration bounds, based on the seed parameter.

        :param connection_m: Connection to the database to log to.
        :param connection_o: Connection to the database holding the observed frequencies to compare to.
        :param uid_observed: IDs of observed samples to compare to.
        :param locus_observed: Loci of observed samples (must match with uid).
        :param simulation_n: Number of simulations to use to obtain a distance.
        :param iterations_n: Number of iterations to run MCMC for.
        :param flush_n: Number of iterations to run MCMC before flushing to disk.
        :param epsilon: Maximum acceptance value for distance between [0, 1].
        :param pi_epsilon: Distribution parameters associated with the walk function.
        :param theta_0: An initial state for the MCMC. If not specified, we pull the last recorded parameter set.
        """
        super().__init__(connection_m, connection_o, uid_observed, locus_observed)

        # Save the parameters associated with our ABC-MCMC itself.
        self.simulation_n, self.iterations_n, self.epsilon = simulation_n, iterations_n, epsilon

        # Create our tables to log to, and save our logging parameter.
        self._create_tables()
        self.flush_n = flush_n

        # Determine our starting point, and save the walk distribution parameters.
        self.theta_0 = theta_0 if theta_0 is not None else self._retrieve_last()
        self.pi_epsilon = pi_epsilon

        # Determine the iterations the MCMC will run for.
        self.boundaries = self._determine_boundaries(iterations_n, theta_0 is None)

    def _log_states(self, x: List, i: int) -> None:
        """ Record our states to our logging database.

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

    def _cleanup(self) -> None:
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
            self._log_states(x, i)

        # Remove our initial guess.
        self._cleanup()
