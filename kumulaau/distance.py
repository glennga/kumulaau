#!/usr/bin/env python3
from kumulaau.observed import tuples_to_pool, tuples_to_sparse_matrix
from numpy import ndarray, dot, arccos, pi, zeros, mean
from typing import List, Callable, Sequence
from types import SimpleNamespace
from argparse import Namespace
from numpy.linalg import norm
from numba import jit


@jit(nopython=True, nogil=True, target='cpu', parallel=True)
def cosine_delta(sample_g: ndarray, observation: ndarray, bounds: ndarray) -> float:
    """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
    determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
    appropriate size and must be zeroed out before use. In order to transform this into a proper distance, we
    compute the angular cosine distance. We assume both vectors are always positive. Optimized by Numba.

    :param sample_g: Generated sample vector, which holds the sampled simulated population.
    :param observation: Observed frequency sample as a sparse frequency vector indexed by repeat length.
    :param bounds: Lower and upper bound (in that order) of the repeat unit space.
    :return: The distance between the generated and observed population.
    """
    kappa, omega = bounds  # Unpack our bounds.

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


@jit(nopython=True, nogil=True, target='cpu', parallel=True)  # TODO: Actually test this metric...
def euclidean_delta(sample_g: ndarray, observation: ndarray, bounds: ndarray) -> float:
    """ Given individuals from the simulated population and the frequencies of individuals from an observed sample,
    determine the differences in distribution for each different simulated sample. All vectors passed MUST be of
    appropriate size and must be zeroed out before use. Treating each distribution as a point, we compute the
    Euclidean distance between both points. Optimized by Numba.

    :param sample_g: Generated sample vector, which holds the sampled simulated population.
    :param observation: Observed frequency sample as a sparse frequency vector indexed by repeat length.
    :param bounds: Lower and upper bound (in that order) of the repeat unit space.
    :return: The distance between the generated and observed population.
    """
    kappa, omega = bounds  # Unpack our bounds.

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


def generate_hdo(observations: Sequence, simulation_n: int, bounds: Sequence) -> SimpleNamespace:
    """ Generate three matrices of interest, we do not populate H and D here.

    H - Binary match matrix of generated x observation. 1 indicates a match, 0 indicates no match.
    D - Distance matrix of generated x observation. Indicates the distance between a given generated-observation entry.
    O - Observation sparse matrix of repeat length x observation, with cell being frequency of length-observation entry.

    :param observations: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :param simulation_n: Number of simulations to run per generated population (number of rows for matched matrix).
    :param bounds: Upper and lower bound (in that order) of the repeat unit space.
    :return: Namespace holding the H, D, and O matrices, as well as the observations and bounds used.
    """
    # Generate the match and distance matrix.
    h = zeros((simulation_n, len(observations)), dtype='int8')
    d = zeros((simulation_n, len(observations)), dtype='float64')

    # Generate the sparse matrix from our observations.
    sparse_matrix = tuples_to_sparse_matrix(observations, bounds)

    return SimpleNamespace(h=h, d=d, o=sparse_matrix, observations=observations, bounds=bounds)


def populate_hd(hdo: SimpleNamespace, sample: Callable, delta: Callable, theta_proposed,
                epsilon: float) -> SimpleNamespace:
    """ Compute the expected distance for all observations to a model generated by our proposed parameter set.
    If the distance between a observation and generated sample falls below epsilon, we count this as a matched
    (marked 1 in the matrix H). Otherwise, we count this as a zero.

    :param hdo: Resultant from a generate_hdo call.
    :param sample: Function such that a population is produced with some parameter set and common ancestor.
    :param delta: Frequency distribution distance function. 0 = exact match, 1 = maximally dissimilar.
    :param theta_proposed: The parameters associated with this matrix instance.
    :param epsilon: The minimum distance between frequencies to label as a match.
    :return: Namespace holding the expected distance for all observations to a population generated by some theta, the
        resulting H matrix, and the resulting D matrix.
    """
    from numpy import array

    # Generate all of our populations and save the generated data we are to compare to (bottleneck is here!!).
    sample_all = array([sample(theta_proposed, _choose_ell_0(hdo.observations, theta_proposed.kappa,
                                                             theta_proposed.omega)) for _ in range(hdo.h.shape[0])])

    # Populate the H and D matrices.
    expected_delta = _delta_matrix(epsilon, sample_all, hdo.o, hdo.h, hdo.d, array(hdo.bounds, dtype='int'), delta)

    # Return our results.
    return SimpleNamespace(expected_delta=expected_delta, h=hdo.h, d=hdo.d)


def likelihood_from_h(h: ndarray) -> float:
    """ 'j' specifies the column or associated observed microsatellite sample in the matched matrix. To determine
    the probability of a model (parameters) matching this observed sample, we compute the average of this column.
    Repeat this for all 'j's, and take the product (assumes each is independent).

    :param h: Matrix of instances where a observation - generated match and don't (matched matrix).
    :return: The likelihood the generated sample set matches our observations.
    """
    from numpy import log, exp

    # Avoid floating point error, use logarithms. Avoid log(0) errors.
    return exp(sum(map(lambda a: 0 if a == 0 else log(a), mean(h, axis=0))))


def _choose_ell_0(observations: Sequence, kappa: int, omega: int) -> List:
    """ We treat the starting repeat length ancestor as a nuisance parameter. We randomly choose a repeat length
    from our observed samples. If this choice exceeds our bounds, we choose our bounds instead.

    :param observations: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :param kappa: Lower bound of our repeat length space.
    :param omega: Upper bound of our repeat length space.
    :return: A single repeat length, wrapped in a list.
    """
    from numpy.random import choice

    return [min(omega, max(kappa, choice(tuples_to_pool(observations))))]


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
    :param bounds: Lower and upper bound (in that order) of the repeat unit space.
    :param delta: Frequency distribution distance function. 0 = exact match, 1 = maximally dissimilar.
    :return: The expected distance for all observations to a population generated by some theta.
    """
    # Iterate through all generated samples.
    for i in range(h.shape[0]):
        for j in range(observations.shape[0]):
            # If our distance is less than a defined epsilon, we mark this as 'matched' with 1. Otherwise, 0.
            d[i, j] = delta(sample_all[i], observations[j], bounds)
            h[i, j] = 1 if d[i, j] < epsilon else 0

    # We return the expected distance for all observations to a population generated by some theta.
    return mean(d)


def get_arguments() -> Namespace:
    """ Create the CLI and parse the arguments, if used as our main script.

    :return: Namespace of all values.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Sample a simulated population and compare this to an observed data set.')
    list(map(lambda a: parser.add_argument(a[0], help=a[1], type=a[2], default=a[3], choices=a[4]), [
        ['-odb', 'Location of the observed database file.', str, 'data/observed.db', None],
        ['-function', 'Distance function to use.', str, None, ['cosine', 'euclidean']],
        ['-uid_observed', 'ID of the observed sample to compare to.', str, None, None],
        ['-locus_observed', 'Locus of the observed sample to compare to.', str, None, None]
    ]))

    return parser.parse_args()


if __name__ == '__main__':
    from kumulaau.observed import extract_alfred_tuples
    from timeit import default_timer as timer
    from kumulaau.model import trace, evolve
    from importlib import import_module
    from numpy import array

    arguments = get_arguments()  # Parse our arguments.

    # Collect observations to compare to.
    main_observed = extract_alfred_tuples([[arguments.uid_observed, arguments.locus_observed]], arguments.odb)

    # Determine our delta and sampling functions.
    main_delta_function = getattr(import_module('kumulaau.distance'), arguments.function + '_delta')
    main_sampler = lambda a, b: evolve(trace(a.n, a.f, a.c, a.d, a.kappa, a.omega), b)

    # A quick and dirty function to generate and populate the HD matrices.
    def main_generate_and_fill_hdo():
        main_hdo = generate_hdo(main_observed, 1000, [3, 30])
        return populate_hd(main_hdo, main_sampler, main_delta_function,
                           SimpleNamespace(n=100, f=100.0, c=0.01, d=0.001, kappa=3, omega=30), 0.1)

    # Run once to remove compile time in elapsed time.
    main_generate_and_fill_hdo()

    start_t = timer()
    main_results = main_generate_and_fill_hdo()  # Execute the sampling and print the running time.
    end_t = timer()
    print('Time Elapsed (1000x): [\n\t' + str(end_t - start_t) + '\n]')

    # Display our results to console, and record to our simulated database.
    print('Expected Distance: [\n\t' + str(main_results.expected_delta) + '\n]')
    print('Likelihood: [\n\t' + str(likelihood_from_h(main_results.h)) + '\n]')
