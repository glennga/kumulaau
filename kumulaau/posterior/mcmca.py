#!/usr/bin/env python3
from kumulaau.distance import generate_hdo, populate_hd, likelihood_from_h
from typing import Callable, Sequence


# The model SQL associated with model database.
SQL = 'TIME_R TIMESTAMP, WAITING_TIME INT, P_PROPOSED FLOAT, EXPECTED_DELTA FLOAT, PROPOSED_TIME INT'


def run(walk: Callable, sample: Callable, delta: Callable, log_handler: Callable, theta_0, observed: Sequence,
        epsilon: float, boundaries: Sequence) -> None:
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

    :param walk: Function that accepts some parameter set and returns another parameter set.
    :param sample: Function that produces a collection of repeat lengths (i.e. the model function).
    :param delta: Function that computes the distance between the result of a sample and an observation.
    :param log_handler: Function that handles what occurs with the current Markov chain and results.
    :param theta_0: Initial starting point.
    :param observed: 2D list of (int, float) tuples representing the (repeat length, frequency) tuples.
    :param epsilon: Maximum acceptance value for distance between [0, 1].
    :param boundaries: Starting and ending iteration for this specific MCMC run.
    :return: None.
    """
    from types import SimpleNamespace
    from numpy.random import uniform
    from datetime import datetime

    # Save our results according to the namespace below.
    a_record = lambda a, b, c, d, e, f: SimpleNamespace(theta=a, time_r=b, waiting_time=c,
                                                        p_proposed=d, expected_delta=e, proposed_time=f)

    # Seed our Markov chain with our initial guess.
    x = [a_record(theta_0, 0, 1, 1.0e-10, 1.0e-10, 0)]

    for i in range(boundaries[0] + 1, boundaries[1]):
        # Walk from our previous state.
        theta_proposed = walk(x[-1].theta)

        # Populate our H, D, and O matrices.
        hdo = generate_hdo(observed, boundaries[1] - boundaries[0], [theta_proposed.kappa, theta_proposed.omega])
        hd_results = populate_hd(hdo, sample, delta, theta_proposed, epsilon)

        # Accept our proposal according to our alpha value.
        p_proposed, p_k = likelihood_from_h(hd_results.h), x[-1].p_proposed
        if p_proposed / p_k > uniform(0, 1):
            x = x + [a_record(theta_proposed, datetime.now(), 1, p_proposed, hd_results.expected_delta, i)]

        # Reject our proposal. We keep our current state and increment our waiting times.
        else:
            x[-1].waiting_time += 1

        # We record to our chain. This is dependent on the current iteration of MCMC.
        log_handler(x, i)
