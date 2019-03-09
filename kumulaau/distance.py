#!/usr/bin/env python3
from numpy import ndarray, dot, arccos, pi, zeros, mean
from abc import ABC, abstractmethod
from typing import List, Callable
from argparse import Namespace
from numpy.linalg import norm
from numba import jit, prange
from sqlite3 import Cursor


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
    def __init__(self, sql_observed: List, kappa: int, omega: int, simulation_n: int):
        """ Constructor. Store the population we are comparing to.

        :param sql_observed: Raw observed frequency samples. Needs to be transformed into sparse vector.
        :param kappa: Lower bound of our state space. omega - kappa = vector dimensionality to extract distance from.
        :param omega: Upper bound of our state space. omega - kappa = vector dimensionality to extract distance from.
        :param simulation_n: Number of simulations to run per generated population (number of rows for matched matrix).
        """
        from numpy import array

        # Save our parameters.
        self.omega, self.kappa, self.simulation_n = omega, kappa, simulation_n

        # Generate the match and distance matrix, and clean the observation set.
        self.observations, self.h, self.d = [array([]) for _ in range(3)]
        self._prepare(sql_observed)

        #  Create a list of all observed lengths.
        self.ell_0_pool = array(list(set([int(b[0]) for a in sql_observed for b in a])))

    def _prepare(self, sql_observed: List) -> None:
        """ Prepare our matched matrix and convert our list of observations into a matrix of sparse frequencies.

        :param sql_observed: A dictionary mapping repeat lengths to frequencies. This is format returned by SQLite.
        :return: None.
        """
        from numpy import array

        # Cast our observed frequencies into numbers.
        observation_dictionary = [{int(a[0]): float(a[1]) for a in b} for b in sql_observed]

        # Fit our observed distribution into a sparse frequency vector.
        self.observations = array([zeros(self.omega - self.kappa + 1) for _ in sql_observed])
        for j, observation in enumerate(observation_dictionary):
            for repeat_unit in observation.keys():
                self.observations[j, repeat_unit - self.kappa + 1] = observation[repeat_unit]

        # Construct an empty matched matrix and deltas matrix.
        self.h = zeros((self.simulation_n, len(sql_observed)), dtype='int8')
        self.d = zeros((self.simulation_n, len(sql_observed)), dtype='float64')

    def _choose_ell_0(self, kappa: int, omega: int) -> List:
        """ We treat the starting repeat length ancestor as a nuisance parameter. We randomly choose a repeat length
        from our observed samples. If this choice exceeds our bounds, we choose our bounds instead.

        :param kappa: Lower bound of our repeat length space.
        :param omega: Upper bound of our repeat length space.
        :return: A single repeat length, wrapped in a list.
        """
        from numpy.random import choice

        return [min(omega, max(kappa, choice(self.ell_0_pool)))]

    @staticmethod
    @abstractmethod
    def _delta(sample_g: ndarray, observation: ndarray, bounds: ndarray) -> float:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
        appropriate size and must be zeroed out before use.

        :param sample_g: Generated sample vector, which holds the sampled simulated population.
        :param observation: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param bounds: Upper and lower bound (in that order) of the repeat unit space.
        :return: The distance between the generated and observed population.
        """
        raise NotImplementedError

    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _delta_matrix(epsilon: float, sample_all: ndarray, observations: ndarray, h: ndarray, d: ndarray,
                      bounds: ndarray, delta: Callable) -> float:
        """ Compute the expected distance for all observations to a set of populations. If the distance between a
        observation and generated sample falls below epsilon, we count this as a matched (marked 1 in the matrix H).
        Otherwise, we count this as a zero. Optimized by Numba.

        :param epsilon: The minimum distance between frequencies to label as a match.
        :param sample_all: Generated sample vector, which holds the sampled simulated population.
        :param observations: Sparse frequency matrix of observations: column = repeat length, row = observation.
        :param h: Matrix of instances where a observation - generated match and don't (matched matrix).
        :param d: Matrix of distances: column = observation, row = generated data.
        :param bounds: Upper and lower bound (in that order) of the repeat unit space.
        :param delta: Frequency distribution distance function. 0 = exact match, 1 = maximally dissimilar.
        :return: The expected distance for all observations to a population generated by some theta.
        """
        # Iterate through all generated samples.
        for i in prange(h.shape[0]):
            for j in range(observations.shape[0]):
                # If our distance is less than a defined epsilon, we mark this as 'matched' with 1. Otherwise, 0.
                d[i, j] = delta(sample_all[i], observations[j], bounds)
                h[i, j] = 1 if d[i, j] < epsilon else 0

        # We return the expected distance for all observations to a population generated by some theta.
        return mean(d)

    def fill_matrices(self, sample: Callable, theta_proposed, epsilon: float) -> float:
        """ Compute the expected distance for all observations to a model generated by our proposed parameter set.
        If the distance between a observation and generated sample falls below epsilon, we count this as a matched
        (marked 1 in the matrix H). Otherwise, we count this as a zero. This function must be called before
        'match_likelihood' as this populates the match matrix.

        :param sample: Function such that a population is produced with some parameter set and common ancestor.
        :param theta_proposed: The parameters associated with this matrix instance.
        :param epsilon: The minimum distance between frequencies to label as a match.
        :return: The expected distance for all observations to a population generated by some theta.
        """
        from numpy import array

        # Generate all of our populations and save the generated data we are to compare to.
        sample_all = array([sample(theta_proposed, self._choose_ell_0(theta_proposed.kappa, theta_proposed.omega))
                            for _ in range(self.h.shape[0])])

        # We are computing the matches and returning the expected distance.
        return self._delta_matrix(epsilon, sample_all, self.observations, self.h, self.d,
                                  array([self.omega, self.kappa], dtype='int'), self._delta)

    def match_likelihood(self) -> float:
        """ 'j' specifies the column or associated observed microsatellite sample in the matched matrix. To determine
        the probability of a model (parameters) matching this observed sample, we compute the average of this column.
        Repeat this for all 'j's, and take the product (assumes each is independent).

        :return: The likelihood the generated sample set matches our observations.
        """
        from numpy import log, exp

        # Avoid floating point error, use logarithms.
        return exp(sum(map(log, mean(self.h, axis=0))))


class Cosine(Distance):
    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _delta(sample_g: ndarray, observation: ndarray, bounds: ndarray) -> float:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
        appropriate size and must be zeroed out before use. In order to transform this into a proper distance, we
        compute the angular cosine distance. We assume both vectors are always positive. Optimized by Numba.

        :param sample_g: Generated sample vector, which holds the sampled simulated population.
        :param observation: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param bounds: Upper and lower bound (in that order) of the repeat unit space.
        :return: The distance between the generated and observed population.
        """
        omega, kappa = bounds  # Unpack our bounds.

        # Prepare the storage vector for our generated frequency vector.
        generated = zeros(omega - kappa + 1)

        # Fit the simulated population into a sparse vector of frequencies.
        for repeat_unit in range(kappa, omega + 1):
            ell_count = 0
            for ell in sample_g:  # Ugly code, but I'm trying to avoid dynamic memory allocation. ):
                ell_count += 1 if ell == repeat_unit else 0
            generated[repeat_unit - kappa] = ell_count / float(sample_g.size)

        # Determine the angular distance. 0 = identical, 1 = maximally dissimilar.
        return 2.0 * arccos(dot(generated, observation) / (norm(generated) * norm(observation))) / pi


class Euclidean(Distance):  # TODO: Actually test this metric...
    @staticmethod
    @jit(nopython=True, nogil=True, target='cpu', parallel=True)
    def _delta(sample_g: ndarray, observation: ndarray, bounds: ndarray) -> float:
        """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
        determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
        appropriate size and must be zeroed out before use. Treating each distribution as a point, we compute the
        Euclidean distance between both points. Optimized by Numba.

        :param sample_g: Generated sample vector, which holds the sampled simulated population.
        :param observation: Observed frequency sample as a sparse frequency vector indexed by repeat length.
        :param bounds: Upper and lower bound (in that order) of the repeat unit space.
        :return: The distance between the generated and observed population.
        """
        omega, kappa = bounds  # Unpack our bounds.

        # Prepare the storage vector for our generated frequency vector.
        generated = zeros(omega - kappa + 1)

        # Fit the simulated population into a sparse vector of frequencies.
        for repeat_unit in range(kappa, omega + 1):
            ell_count = 0
            for ell in sample_g:  # Ugly code, but I'm trying to avoid dynamic memory allocation. ):
                ell_count += 1 if ell == repeat_unit else 0
            generated[repeat_unit - kappa] = ell_count / float(sample_g.size)

        # Determine the Euclidean distance. 0 = identical, 1 = maximally dissimilar.
        return norm(generated - observation)


def get_arguments() -> Namespace:
    """ Create the CLI and parse the arguments, if used as our main script.

    :return: Namespace of all values.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Sample a simulated population and compare this to an observed data set.')
    list(map(lambda a: parser.add_argument(a[0], help=a[1], type=a[2], default=a[3], choices=a[4]), [
        ['-odb', 'Location of the observed database file.', str, 'data/observed.db', None],
        ['-rdb', 'Location of the database to record data to.', str, 'data/delta.db', None],
        ['-function', 'Distance function to use.', str, None, ['COSINE', 'EUCLIDEAN']],
        ['-uid_observed', 'ID of the observed sample to compare to.', str, None, None],
        ['-locus_observed', 'Locus of the observed sample to compare to.', str, None, None]
    ]))

    return parser.parse_args()


if __name__ == '__main__':
    from timeit import default_timer as timer
    from numpy import array
    from sqlite3 import connect
    import pop

    main_arguments = get_arguments()  # Parse our arguments.

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

    # Create our parameter sets and sampling lambda.
    main_theta = {'n': 100, 'f': 100.0, 'c': 0.01, 'd': 0.001, 'kappa': 3, 'omega': 30}
    main_accumulator_parameters = [[main_observed_frequency], 3, 30, 1000]
    main_accumulator = Cosine(*main_accumulator_parameters) if main_arguments.function == 'COSINE' \
        else Euclidean(*main_accumulator_parameters)
    sampler = lambda a, b: pop.evolve(pop.trace(a.n, a.f, a.c, a.d, a.kappa, a.omega), b)

    # Execute the sampling and print the running time.
    start_t = timer()
    expected_delta = main_accumulator.fill_matrices(sampler, Namespace(**main_theta), 0.1)
    end_t = timer()
    print('Time Elapsed (1000x): [\n\t' + str(end_t - start_t) + '\n]')

    # Display our results to console, and record to our simulated database.
    print('Expected Distance: [\n\t' + str(expected_delta) + '\n]')
    print('Likelihood: [\n\t' + str(main_accumulator.match_likelihood()) + '\n]')
    log_distances(cursor_r, array([expected_delta]), main_arguments.uid_observed, main_arguments.locus_observed)
    connection_r.commit(), connection_r.close(), connection_o.close()
