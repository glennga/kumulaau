#!/usr/bin/env python3
from sqlite3 import Cursor
from typing import List
from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import ndarray, dot, arccos, pi
from numba import jit


def create_table(cursor: Cursor) -> None:
    """ Create the table to log the results of our comparisons to.

    :param cursor: Cursor to the database file to log to.
    :return: None.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS DISTANCE_POP (
            TIME_R TIMESTAMP,
            ID_GENERATED TEXT,
            UID_OBSERVED TEXT,
            LOCUS_OBSERVED TEXT,
            DISTANCE FLOAT
        );""")


def log_distances(cursor: Cursor, distances: ndarray, uid_observed: str, locus_observed: str) -> None:
    """ Given the computed differences between two sample's distributions, record each with a unique ID into the
     database.

    :param cursor: Cursor to the database to log to.
    :param distances: Computed differences from the sampling.
    :param uid_observed: ID of the observed sample to compare to.
    :param locus_observed: Locus of the observed sample to compare to.
    :return: None.
    """
    from string import ascii_uppercase, digits
    from datetime import datetime
    from random import choice

    # Record our results.
    cursor.executemany("""
        INSERT INTO DISTANCE_POP
        VALUES (?, ?, ?, ?, ?)
    """, ((datetime.now(), ''.join(choice(ascii_uppercase + digits) for _ in range(20)), uid_observed,
           locus_observed, a) for a in distances))


class Distance(ABC):
    def __init__(self, observed_frequencies_dirty: List, kappa: int, omega: int, simulation_n: int,
                 generated_sample: ndarray=None):
        """ Constructor. Store the population we are comparing to.

        :param observed_frequencies_dirty: Dirty observed frequency samples. Needs to be transformed into sparse vector.
        :param kappa: Lower bound of our state space. omega - kappa = vector dimensionality to extract distance from.
        :param omega: Upper bound of our state space. omega - kappa = vector dimensionality to extract distance from.
        :param simulation_n: Number of simulations to run per generated population (number of rows for matched matrix).
        :param generated_sample: Simulated population of repeat units (one dimensional list). Specify now or later.
        """
        from numpy import array

        # Save our parameters.
        self.generated_sample, self.omega, self.kappa = generated_sample, omega, kappa

        # We will set the following fields upon preparation.
        self.generated_frequency, self.observed_frequencies, self.matched_matrix = [array([]) for _ in range(3)]

        # Generate the match matrix and clean the observation set.
        self._prepare(observed_frequencies_dirty, simulation_n)

    def _prepare(self, observed_frequencies_dirty: List, simulation_n: int) -> None:
        """ TODO

        :param observed_frequencies_dirty:
        :param simulation_n:
        :return:
        """
        from numpy import zeros

        # Cast our observed frequencies into numbers.
        observation_dictionary = [{int(a[0]): float(a[1]) for a in b} for b in observed_frequencies_dirty]

        # Fit our observed distribution into a sparse frequency vector.
        self.observed_frequencies = [zeros(self.omega - self.kappa + 1) for _ in observed_frequencies_dirty]
        for j, observation in enumerate(observation_dictionary):
            for repeat_unit in observation.keys():
                self.observed_frequencies[j][repeat_unit - self.kappa + 1] = observation[repeat_unit]

        # Prepare the storage vector for our generated frequency vector.
        self.generated_frequency = zeros(self.omega - self.kappa + 1)

        # Construct an empty matched matrix.
        self.matched_matrix = zeros((simulation_n, len(observed_frequencies_dirty)), dtype='bool')

    @staticmethod
    @abstractmethod
    def _delta(generated_sample: ndarray, generated_frequency: ndarray, observed_frequency: ndarray,
               omega: int, kappa: int) -> float:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
        appropriate size and must be zeroed out before use. Optimized by Numba.

        :param generated_sample: Storage vector, used to hold the sampled simulated population.
        :param generated_frequency: Storage sparse vector, used to hold the frequency sample.
        :param observed_frequency: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param omega: Upper bound of the repeat unit space.
        :param kappa: Lower bound of the repeat unit space.
        :return: The distance between the generated and observed population.
        """
        raise NotImplementedError

    def _compute_delta(self, observed_frequency: ndarray, generated_sample: ndarray) -> float:
        """ Compute the distance between a generated sample and an observation.

        :param observed_frequency: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param generated_sample: Simulated population of repeat units (one dimensional list). Optional.
        :return: The computed distance.
        """
        return self._delta(generated_sample, self.generated_frequency, observed_frequency, self.omega, self.kappa)

    def compute_deltas(self, epsilon: float, generated_number: int, generated_sample: ndarray=None) -> ndarray:
        """ Compute the distance between a set of observed frequencies and some generated sample. If a given
        distance falls below epsilon, we mark this as a match (1 in matched_matrix). Otherwise, it gets marked as a 0.

        :param epsilon: The minimum distance between frequencies to label as a match.
        :param generated_number: The current generated population we are working with. Indicates the row to access.
        :param generated_sample: Simulated population of repeat units (one dimensional list). Optional.
        :return: The vector of distances from each observed frequency to the generated frequency.
        """
        from numpy import zeros

        # For each of observed frequencies, we compute a distance.
        self.generated_sample = generated_sample if generated_sample is not None else self.generated_sample
        deltas_at_number = zeros(len(self.observed_frequencies))

        # If our distance is less than a defined epsilon, we mark this as 'matched' with 1. Otherwise, 0.
        for j, observed_frequency in enumerate(self.observed_frequencies):
            deltas_at_number[j] = self._compute_delta(observed_frequency, self.generated_sample)
            self.matched_matrix[generated_number, j] = deltas_at_number[j] < epsilon

        # Return our distance vector.
        return deltas_at_number

    def match_likelihood(self) -> float:
        """ 'j' specifies the column or associated observed microsatellite sample in 'matched_matrix'. To determine
        the probability of a model (parameters) matching this observed sample, we compute the average of this column.
        Repeat this for all 'j's, and take the product (assumes each is independent).

        :return: The likelihood the generated sample set matches our observations.
        """
        from numpy import mean, log, exp

        # Avoid floating point error, use logarithms.
        return exp(sum(map(log, mean(self.matched_matrix, axis=0))))


class Cosine(Distance):
    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _delta(generated_sample: ndarray, generated_frequency: ndarray, observed_frequency: ndarray,
               omega: int, kappa: int) -> float:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
        appropriate size and must be zeroed out before use. In order to transform this into a proper distance, we
        compute the angular cosine distance. We assume both vectors are always positive. Optimized by Numba.

        :param generated_sample: Storage vector, used to hold the sampled simulated population.
        :param generated_frequency: Storage sparse vector, used to hold the frequency sample.
        :param observed_frequency: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param omega: Upper bound of the repeat unit space.
        :param kappa: Lower bound of the repeat unit space.
        :return: The distance between the generated and observed population.
        """
        # Fit the simulated population into a sparse vector of frequencies.
        for repeat_unit in range(kappa, omega + 1):
            i_count = 0
            for i in generated_sample:  # Ugly code, but I'm trying to avoid dynamic memory allocation. ):
                i_count += 1 if i == repeat_unit else 0
            generated_frequency[repeat_unit - kappa] = i_count / float(generated_sample.size)

        # Determine the angular distance. 0 = identical, 1 = maximally dissimilar.
        return 2.0 * arccos(dot(generated_frequency, observed_frequency) /
                            (norm(generated_frequency) * norm(observed_frequency))) / pi


class Euclidean(Distance):  # TODO: Actually test this metric...
    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _delta(generated_sample: ndarray, generated_frequency: ndarray, observed_frequency: ndarray,
               omega: int, kappa: int) -> float:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
        appropriate size and must be zeroed out before use. Treating each distribution as a point, we compute the
        Euclidean distance between both points. Optimized by Numba.

        :param generated_sample: Storage vector, used to hold the sampled simulated population.
        :param generated_frequency: Storage sparse vector, used to hold the frequency sample.
        :param observed_frequency: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param omega: Upper bound of the repeat unit space.
        :param kappa: Lower bound of the repeat unit space.
        :return: The distance between the generated and observed population.
        """
        # Fit the simulated population into a sparse vector of frequencies.
        for repeat_unit in range(kappa, omega + 1):
            i_count = 0
            for i in generated_sample:  # Ugly code, but I'm trying to avoid dynamic memory allocation. ):
                i_count += 1 if i == repeat_unit else 0
            generated_frequency[repeat_unit - kappa] = i_count / float(generated_sample.size)

        # Determine the Euclidean distance. 0 = identical, 1 = maximally dissimilar.
        return norm(generated_frequency - observed_frequency)


if __name__ == '__main__':
    from population import Population, BaseParameters
    from argparse import ArgumentParser
    from numpy import zeros, array
    from sqlite3 import connect

    parser = ArgumentParser(description='Sample a simulated population and compare this to an observed data set.')
    parser.add_argument('-odb', help='Location of the observed database file.', type=str, default='data/observed.db')
    parser.add_argument('-rdb', help='Location of the database to record data to.', type=str, default='data/delta.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    parser.add_argument('-function', help='Similarity function to use.', type=str, choices=['COSINE'])
    paa('-uid_observed', 'ID of the observed sample to compare to.', str)
    paa('-locus_observed', 'Locus of the observed sample to compare to.', str)
    main_arguments = parser.parse_args()  # Parse our arguments.

    # Connect to all of our databases, and create our table if it does not already exist.
    connection_o, connection_r = connect(main_arguments.odb), connect(main_arguments.rdb)
    cursor_o, cursor_r = connection_o.cursor(), connection_r.cursor()
    create_table(cursor_r)

    main_observed_frequency = connection_o.execute(""" -- Pull the frequency from the observed database. --
        SELECT ELL, ELL_FREQ
        FROM OBSERVED_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (main_arguments.uid_observed, main_arguments.locus_observed,)).fetchall()

    # Generate some population.
    main_population = Population(BaseParameters(n=100, f=1.0, c=0.01, u=1.2,
                                                d=0.001, kappa=3, omega=30)).evolve(array([11]))

    # Execute the sampling.
    main_distances = Cosine([main_observed_frequency], 3, 30, 1).compute_deltas(1.0, 0, main_population)

    # Display our results to console, and record to our simulated database.
    print('Result: [\n\t' + ', '.join(str(a) for a in main_distances) + '\n]')
    log_distances(cursor_r, main_distances, main_arguments.uid_observed, main_arguments.locus_observed)
    connection_r.commit(), connection_r.close(), connection_o.close()
