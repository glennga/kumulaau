#!/usr/bin/env python3
from kumulaau.distance import populate_d
from typing import Callable, Sequence
from numpy import ndarray


def _populate_h(h: ndarray, d: ndarray, epsilon: float) -> None:
    """ If the distance between a observation and generated sample falls below epsilon, we count this as a matched
    (marked 1 in the matrix H). Otherwise, we count this as a zero. This is the ABC portion.

    :param h: Binary match matrix of generated x observation to populate. Must be the same size as D.
    :param d: **Populated** D matrix, holding all distances between a generated and observed population.
    :param epsilon: The minimum distance between frequencies to label as a match.
    :return: None.
    """
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            h[i, j] = 1 if d[i, j] < epsilon else 0


def _likelihood_from_h(h: ndarray) -> float:
    """ 'j' specifies the column or associated observed microsatellite sample in the matched matrix. To determine
    the probability of a model (parameters) matching this observed sample, we compute the average of this column.
    Repeat this for all 'j's, and take the product (assumes each is independent).

    :param h: Matrix of instances where a observation - generated match and don't (matched matrix).
    :return: The likelihood the generated sample set matches our observations.
    """
    from numpy import log, exp, mean

    # Avoid floating point error, use logarithms. Avoid log(0) errors.
    col_sum = sum(map(lambda a: 0 if a == 0 else log(a), mean(h, axis=0)))
    return 0 if col_sum == 0 else exp(col_sum)


def run(walk: Callable, sample: Callable, delta: Callable, log_handler: Callable, theta_0, observed: Sequence,
        simulation_n: int, boundaries: Sequence, epsilon: float) -> None:
    """ A MCMC algorithm to approximate the posterior distribution of a generic model, whose acceptance to the
    chain is determined by some distance between repeat length distributions. My interpretation of this
    ABC-MCMC approach is given below:

    1) We start with some initial guess theta_0. Right off the bat, we move to another theta from theta_0.
    2) For 'boundaries[1] - boundaries[0]' iterations...
        a) For 'simulation_n' iterations...
            i) We simulate a population using the given theta.
            ii) For each observed frequency ... 'D'
                1) We compute the difference between the two distributions.
                2) If this difference is less than our epsilon term, we add 1 to a vector modeling D.
        b) Compute the probability that each observed matches a generated population: all of D / simulation_n.
        c) If this probability is greater than the probability of the previous, we accept.
        d) Otherwise, we accept our proposed with probability p(proposed) / p(prev).

    :param walk: Function that accepts some parameter set and returns another parameter set.
    :param sample: Function that produces a collection of repeat lengths (i.e. the model function).
    :param delta: Function that computes the distance between the result of a sample and an observation.
    :param log_handler: Function that handles what occurs with the current Markov chain and results.
    :param theta_0: Initial starting point.
    :param observed: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :param simulation_n: Number of simulations to use to obtain a distance.
    :param boundaries: Starting and ending iteration for this specific MCMC run.
    :param epsilon: Maximum acceptance value for distance between [0, 1].
    :return: None.
    """
    from types import SimpleNamespace
    from numpy.random import uniform
    from datetime import datetime
    from numpy import zeros, mean

    # Save our results according to the namespace below.
    a_record = lambda a_1, b_1, c_1, d_1, e_1, f_1: SimpleNamespace(theta=a_1, time_r=b_1, waiting_time=c_1,
                                                                    p_proposed=d_1, expected_delta=e_1,
                                                                    proposed_time=f_1)

    # Seed our Markov chain with our initial guess.
    x = [a_record(theta_0, 0, 1, 1.0e-10, 1.0e-10, 0)]

    for i in range(boundaries[0] + 1, boundaries[1]):
        theta_proposed = walk(x[-1].theta)  # Walk from our previous state.

        # Prepare our H and D matrices.
        h = zeros((simulation_n, len(observed)), dtype='int8')
        d = zeros((simulation_n, len(observed)), dtype='float64')

        # Populate D, then H.
        populate_d(d, observed, sample, delta, theta_proposed, [theta_proposed.kappa, theta_proposed.omega])
        _populate_h(h, d, epsilon)

        # Accept our proposal according to our alpha value. Metropolis sampling.
        p_proposed, p_k = _likelihood_from_h(h), x[-1].p_proposed
        if p_proposed / p_k > uniform(0, 1):
            x = x + [a_record(theta_proposed, datetime.now(), 1, p_proposed, mean(d), i)]

        # Reject our proposal. We keep our current state and increment our waiting times.
        else:
            x[-1].waiting_time += 1

        # We record to our chain. This is dependent on the current iteration of MCMC.
        log_handler(x, i)
