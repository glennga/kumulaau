#!/usr/bin/env python3
from sqlite3 import Cursor
from typing import List
from numpy.linalg import norm
from numpy import ndarray, dot
from numba import jit, prange
from abc import ABC, abstractmethod


def create_table(cursor: Cursor) -> None:
    """ Create the table to log the results of our comparisons to.

    :param cursor: Cursor to the database file to log to.
    :return: None.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS SUMMARY_POP (
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
        INSERT INTO SUMMARY_POP
        VALUES (?, ?, ?, ?, ?)
    """, ((datetime.now(), ''.join(choice(ascii_uppercase + digits) for _ in range(20)), uid_observed,
           locus_observed, a) for a in distances))


class Summary(ABC):
    def __init__(self, generated_sample: ndarray=None):
        """ Constructor. Store the population we are comparing to.

        :param generated_sample: Simulated population of repeat units (one dimensional list). Optional.
        """
        from numpy import array
        self.generated_sample = generated_sample

        # We will set the following fields upon preparation.
        self.generated_frequency, self.observed_frequency, self.distances = [array([]) for _ in range(3)]

    def _prepare(self, observed_frequency_dirty: List) -> None:
        """ Given a "dirty" (non-length normalized) frequency vector of observed samples, generate the appropriate
        vectors to store to.

        :param observed_frequency_dirty: Dirty observed frequency sample. Needs to be transformed into a sparse vector.
        :return: None.
        """
        from numpy import zeros

        # Cast our observed frequencies into numbers.
        observed_dictionary = {int(a[0]): float(a[1]) for a in observed_frequency_dirty}

        # Determine the omega and kappa from the simulated population and our observed sample.
        self.omega = max(self.generated_sample) + 1 if \
            max(self.generated_sample) > max(observed_dictionary.keys()) else max(observed_dictionary.keys()) + 1
        self.kappa = min(self.generated_sample) if \
            min(self.generated_sample) < min(observed_dictionary.keys()) else min(observed_dictionary.keys())

        # Create the vectors to return.
        self.generated_frequency, self.observed_frequency, self.distances = \
            [zeros(self.omega), zeros(self.omega), zeros(1)]

        # Fit our observed distribution into a sparse frequency vector.
        for repeat_unit in observed_dictionary.keys():
            self.observed_frequency[repeat_unit] = observed_dictionary[repeat_unit]

    @staticmethod
    @abstractmethod
    def _distance(generated_sample: ndarray, generated_frequency: ndarray, observed_frequency: ndarray,
                  distances: ndarray, omega: int, kappa: int) -> None:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        sample the same amount from the simulated population 'r' times and determine the differences in distribution
        for each different simulated sample. All vectors passed MUST be of appropriate size and must be zeroed out
        before use. Optimized by Numba.

        :param generated_sample: Storage vector, used to hold the sampled simulated population.
        :param generated_frequency: Storage sparse vector, used to hold the frequency sample.
        :param observed_frequency: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param distances: Output vector, used to store the computed distances of each sample.
        :param omega: Upper bound of the repeat unit space.
        :param kappa: Lower bound of the repeat unit space.
        """
        raise NotImplementedError

    def compute_distance(self, observed_frequency_dirty: List, generated_sample: ndarray=None) -> ndarray:
        """ Compute the distance for 'sample_n' samples, and return the results. For repeated runs of this function,
        we append our results to the same 'distances' field.

        :param observed_frequency_dirty: Dirty observed frequency sample. Needs to be transformed into a sparse vector.
        :param generated_sample: Simulated population of repeat units (one dimensional list). Optional.
        :return: The computed distances for each sample.
        """
        from numpy import concatenate, array
        self.generated_sample = generated_sample if generated_sample is not None else array([])

        d_previous = array(self.distances)  # Save our previous state.
        self._prepare(observed_frequency_dirty)

        # Run our sampling and comparison. Concatenate our previous state.
        self._distance(self.generated_sample, self.generated_frequency, self.observed_frequency,
                       self.distances, self.omega, self.kappa)
        self.distances = concatenate((self.distances, d_previous), axis=None)

        return self.distances

    def compute_distance_multiple(self, observed_frequencies_dirty: List, generated_sample: ndarray=None) -> None:
        """ TODO: Finish documentation for compute_distance_multiple.

        :param observed_frequencies_dirty: Dirty observed frequency samples.
        :param generated_sample: Simulated population of repeat units (one dimensional list). Optional.
        :return:
        """
        for a in observed_frequencies_dirty:
            self.compute_distance(a, generated_sample)

    def average_distance(self):
        """ TODO: Finish documentation for average_distance.

        :return:
        """
        from numpy import average
        return average(self.distances)


class Frequency(Summary):
    def __init__(self, generated_sample: ndarray=None):
        """ Constructor. Store the population we are comparing to.

        :param generated_sample: Simulated population of repeat units (one dimensional list). Optional.
        """
        super(Frequency, self).__init__(generated_sample)
        raise DeprecationWarning  # Do not want to use this class for comparison... Cosine is better.

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _distance(generated_sample: ndarray, generated_frequency: ndarray, observed_frequency: ndarray,
                  distances: ndarray, omega: int, kappa: int) -> None:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        sample the same amount from the simulated population 'r' times and determine the differences in distribution
        for each different simulated sample. All vectors passed MUST be of appropriate size and must be zeroed out
        before use. Optimized by Numba.

        :param generated_sample: Storage vector, used to hold the sampled simulated population.
        :param generated_frequency: Storage sparse vector, used to hold the frequency sample.
        :param observed_frequency: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param distances: Output vector, used to store the computed distances of each sample.
        :param omega: Upper bound of the repeat unit space.
        :param kappa: Lower bound of the repeat unit space.
        :return: None.
        """
        for k_0 in prange(distances.size):
            # Fit the simulated population into a sparse vector of frequencies.
            for repeat_unit in prange(kappa, omega):
                i_count = 0
                for i in generated_sample:  # Ugly code, but I'm trying to avoid memory allocation. ):
                    i_count += 1 if i == repeat_unit else 0
                generated_frequency[repeat_unit] = i_count / generated_sample.size

            # For all repeat lengths, determine the sum difference in frequencies. Normalize this to [0, 1].
            distances[k_0] = 0
            for j in prange(kappa, omega):
                distances[k_0] += abs(generated_frequency[j] - observed_frequency[j])
            distances[k_0] = 1 - (distances[k_0] / 2.0)


class Cosine(Summary):
    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _distance(generated_sample: ndarray, generated_frequency: ndarray, observed_frequency: ndarray,
                  distances: ndarray, omega: int, kappa: int) -> None:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        sample the same amount from the simulated population 'r' times and determine the differences in distribution
        for each different simulated sample. All vectors passed MUST be of appropriate size and must be zeroed out
         before use. Optimized by Numba.

        :param generated_sample: Storage vector, used to hold the sampled simulated population.
        :param generated_frequency: Storage sparse vector, used to hold the frequency sample.
        :param observed_frequency: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param distances: Output vector, used to store the computed distances of each sample.
        :param omega: Upper bound of the repeat unit space.
        :param kappa: Lower bound of the repeat unit space.
        :return: None.
        """
        for k_0 in prange(distances.size):
            # Fit the simulated population into a sparse vector of frequencies.
            for repeat_unit in prange(kappa, omega):
                i_count = 0
                for i in generated_sample:  # Ugly code, but I'm trying to avoid memory allocation. ):
                    i_count += 1 if i == repeat_unit else 0
                generated_frequency[repeat_unit] = i_count / generated_sample.size

            # For all repeat lengths, determine the sum difference in frequencies. Normalize this to [0, 1].
            distances[k_0] = (dot(generated_frequency, observed_frequency) /
                              (norm(generated_frequency) * norm(observed_frequency)))


if __name__ == '__main__':
    from population import Population, BaseParameters
    from argparse import ArgumentParser
    from numpy import zeros, array
    from sqlite3 import connect

    parser = ArgumentParser(description='Sample a simulated population and compare this to an observed data set.')
    parser.add_argument('-odb', help='Location of the observed database file.', type=str, default='data/observed.db')
    parser.add_argument('-rdb', help='Location of the database to record data to.', type=str, default='data/summary.db')
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    parser.add_argument('-function', help='Similarity function to use.', type=str, choices=['COSINE', 'FREQ'])
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
                                                d=0.001, kappa=3, omega=100)).evolve(array([11]))

    # Execute the sampling.
    if main_arguments.function.casefold() == 'freq':
        main_distances = Frequency(main_population).compute_distance(main_observed_frequency)
    else:
        main_distances = Cosine(main_population).compute_distance(main_observed_frequency)

    # Display our results to console, and record to our simulated database.
    print('Result: [\n\t' + ', '.join(str(a) for a in main_distances) + '\n]')
    log_distances(cursor_r, main_distances, main_arguments.uid_observed, main_arguments.locus_observed)
    connection_r.commit(), connection_r.close(), connection_o.close()
