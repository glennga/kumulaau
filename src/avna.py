#!/usr/bin/env python3
from mutate import BaseParameters
from migration import Migration
from model import choose_i_0
from typing import Dict, List
from sqlite3 import Cursor
from numpy import ndarray


class AVNAParameters(BaseParameters):
    def __init__(self, i_0: ndarray, big_n: int, mu: float, s: float, kappa: int, omega: int, u: float, v: float,
                 m: float, p: float, f_afr: float, f_nafr: float, tau: int):
        """ Constructor. This is meant to be a data class for the AVNA model.

        :param i_0: Repeat lengths of the common ancestors.
        :param big_n: Effective population size of the ancestor class, and with 'f' parameters for other populations.
        :param mu: Mutation rate, bounded by (0, infinity).
        :param s: Proportional rate (to repeat length), bounded by (-1 / (omega - kappa + 1), infinity).
        :param kappa: Lower bound of possible repeat lengths (minimum of state space).
        :param omega: Upper bound of possible repeat lengths (maximum of state space).
        :param u: Constant bias parameter, used to determine the probability of an expansion and bounded by [0, 1].
        :param v: Linear bias parameter, used to determine the probability of an expansion.
        :param m: Success probability for the truncated geometric distribution, bounded by [0, 1].
        :param p: Probability that the geometric distribution will be used vs. single repeat length mutations.
        :param f_afr: Scaling factor for the African population.
        :param f_nafr: Scaling factor for the Non-African population.
        :param tau: Period of the African - Non-African merge, relative to when we have to sample.
        """
        super(AVNAParameters, self).__init__(i_0=i_0, big_n=big_n, f=1.0, mu=mu, s=s, kappa=kappa, omega=omega,
                                             u=u, v=v, m=m, p=p)  # The scaling factor for our ancestor is 1.

        # Append the parameters of the AVNA model.
        self.f_afr, self.f_nafr, self.tau = f_afr, f_nafr, tau

        # There exist 13 parameters.
        self.PARAMETER_COUNT = 13

    def __iter__(self):
        """ Return each our parameters in the following order:

         big_n, f, mu, s, kappa, omega, u, v, m, p, f_afr, f_nafr, tau

        :return: Iterator for all of our parameters.
        """
        for parameter in [self.big_n, self.f, self.mu, self.s, self.kappa, self.omega, self.u, self.v, self.m, self.p,
                          self.f_afr, self.f_nafr, self.tau]:
            yield parameter


class AVNA(Migration):
    def __init__(self, j: Dict, afs_d: List, cur_j: Cursor):
        """ Constructor. Load our parameters from the dictionary (from JSON), and ensure that all exist here.

        :param j: Dictionary of configuration parameters describing AVNA.
        """
        j_s, j_r = j['simulation'], j['real-samples']

        self.theta_0 = {  # Define our starting position in our parameter space.
            k: AVNAParameters(i_0=choose_i_0(afs_d), big_n=j_s[k]['big_n_anc'], mu=j_s[k]['mu'],
                              kappa=j_s[k]['kappa'], omega=j_s[k]['omega'], s=j_s[k]['s'], u=j_s[k]['u'],
                              v=j_s[k]['v'], m=j_s[k]['m'], p=j_s[k]['p'], f_afr=j_s[k]['f_afr'],
                              f_nafr=j_s[k]['f_nafr'], tau=j_s[k]['tau']) for k in ['anc', 'afr', 'naf']
        }
        self.theta_0_sigma = {  # Define how far we walk. Next state of chain is distributed by parameters below.
            k: AVNAParameters(i_0=choose_i_0(afs_d), big_n=j_s[k]['sigma']['big_n_anc'], mu=j_s[k]['sigma']['mu'],
                              kappa=j_s[k]['sigma']['kappa'], omega=j_s[k]['sigma']['omega'],
                              s=j_s[k]['sigma']['s'], u=j_s[k]['sigma']['u'], v=j_s[k]['sigma']['v'],
                              m=j_s[k]['sigma']['m'], p=j_s[k]['sigma']['p'], f_afr=j_s[k]['sigma']['f_afr'],
                              f_nafr=j_s['sigma']['f_nafr'], tau=j_s[k]['sigma']['tau']) for k in ['anc', 'afr', 'naf']
        }

        self.rfs_d = {  # Assemble the real frequency sample distributions for both AFR and NAF.
            k: list(map(lambda a, b: cur_j.execute(""" -- Pull the frequency distributions from the real database. --
                SELECT ELL, ELL_FREQ
                FROM REAL_ELL
                WHERE SAMPLE_UID LIKE ?
                AND LOCUS LIKE ?
            """, (a, b,)).fetchall(), j_r[k]['rsu'], j_r[k]['l'])) for k in ['afr', 'naf']
        }
        self.two_little_n = {  # Assemble the sample sizes associated with the samples above for AFR and NAF.
            k: list(map(lambda a, b: int(cur_j.execute(""" -- Retrieve the sample sizes, the number of alleles. --
                SELECT SAMPLE_SIZE
                FROM REAL_ELL
                WHERE SAMPLE_UID LIKE ?
                AND LOCUS LIKE ?
            """, (a, b,)).fetchone()[0]), j_r[k]['rsu'], j_r[k]['l'])) for k in ['afr', 'naf']
        }

        self.afs_d = afs_d  # We save the distribution of all real samples.

    @staticmethod
    def walk(theta: AVNAParameters, theta_sigma: AVNAParameters) -> AVNAParameters:
        """ TODO:

        :param theta:
        :param theta_sigma:
        :return:
        """
        from numpy.random import normal, lognormal
        from numpy import empty

        walk = lambda a, b, c=False: normal(a, b) if c is False else round(normal(a, b))
        walk_mu = lambda a, b: lognormal(a, b)  # With mu, we draw from a log-normal distribution.

        return AVNAParameters(i_0=empty(0), big_n=walk(theta.big_n, theta_sigma.big_n, True),
                              mu=walk_mu(theta.mu, theta_sigma.mu), kappa=walk(theta.kappa, theta_sigma.kappa, True),
                              omega=walk(theta.omega, theta_sigma.omega, True), s=walk(theta.s, theta_sigma.s),
                              u=walk(theta.u, theta_sigma.u), v=walk(theta.v, theta_sigma.v),
                              m=walk(theta.m, theta_sigma.m), p=walk(theta.p, theta_sigma.p),
                              f_afr=walk(theta.f_afr, theta_sigma.f_afr), f_nafr=walk(theta.f_nafr, theta_sigma.f_nafr),
                              tau=walk(theta.tau, theta_sigma.tau))

    def compare(self, z_afr: ndarray, z_naf: ndarray, rs: int):
        """ TODO:

        :param z_afr:
        :param z_naf:
        :param rs:
        :return:
        """
        from compare import prepare_frequency, frequency_delta
        from numpy import average
        summary_deltas_afr, summary_deltas_naf = [], []

        for d in zip(self.rfs_d['afr'], self.two_little_n['afr']):  # Determine the delta term for AFR first.
            scs, sfs, rfs, delta_rs = prepare_frequency(d[0], z_afr, d[1], rs)
            frequency_delta(scs, sfs, rfs, z_afr, delta_rs)
            summary_deltas_afr = summary_deltas_afr + [1 - average(delta_rs)]

        for d in zip(self.rfs_d['naf'], self.two_little_n['naf']):  # Repeat for NAF.
            scs, sfs, rfs, delta_rs = prepare_frequency(d[0], z_naf, d[1], rs)
            frequency_delta(scs, sfs, rfs, z_naf, delta_rs)
            summary_deltas_afr = summary_deltas_afr + [1 - average(delta_rs)]

        return average(summary_deltas_afr + summary_deltas_naf)

    def mcmc(self, it: int, rs: int, rp: int, epsilon: float):
        """ My interpretation of an MCMC-ABC rejection sampling approach to approximate the posterior distribution of
        the AVNA (Africa vs. Non Africa) population migration model. The steps taken are as follows:

        1) We start with some initial guess and simulate individuals with an effective population size of
           set branch size (i.e. we stop at the branch size). We denote this as population ANC.
        2) The individuals of ANC are them randomly used as seeds for two child populations: AFR (Africa) and NAF
           (Non-Africa). Simulate these populations in serial (parallelism involved with simulation one population is
           well used, I think).
        3) Compute the average distance between a set of simulated AFR and real AFR (delta). Repeat for NAF.
           a) If this term is above some defined epsilon, we append this to our chain.
           b) Otherwise, we reject it and increase the waiting time of our current parameter state by one.
        4) Repeat for 'it' iterations.

        Source: https://theoreticalecology.wordpress.com/2012/07/15/
                a-simple-approximate-bayesian-computation-mcmc-abc-mcmc-in-r/

        :param it: Number of iterations to run MCMC for.
        :param rs: Number of samples per simulation to use to obtain delta.
        :param rp: Number of simulations to use to obtain delta.
        :param epsilon: Minimum acceptance value for delta.
        """
        from numpy import setdiff1d
        from immediate import Immediate
        from numpy.random import choice

        states = [[]]  # Seed our chain with our initial guess.

        for j in range(1, it):
            thetas = states[-1][0]  # Our current position in the state space.

            theta_anc_proposed = self.walk(thetas[0], self.theta_0_sigma['anc'])  # Walk from our current position.
            theta_afr_proposed = self.walk(thetas[1], self.theta_0_sigma['afr'])
            theta_naf_proposed = self.walk(thetas[2], self.theta_0_sigma['naf'])

            summary_simulated_delta = 0

            for zp in range(rp):
                theta_anc_proposed.i_0 = choose_i_0(self.afs_d)  # Generate our ANC population.
                z_anc = Immediate(theta_anc_proposed).evolve()

                # Select the ancestors for the AFR and NAF populations. There must exist at-least 1 in each.
                theta_afr_proposed.i_0 = choice(z_anc, size=choice([a for a in range(1, len(z_anc))]))
                theta_naf_proposed.i_0 = setdiff1d(z_anc, theta_afr_proposed.i_0)

                # Generate the AFR and NAF populations.
                z_afr, z_naf = Immediate(theta_afr_proposed).evolve(), Immediate(theta_naf_proposed).evolve()

                # Compute the delta term for our current population set.
                summary_simulated_delta += self.compare(z_afr, z_naf, rs)

            summary_simulated_delta /= rp  # Accept our proposal if the current distance lies above our epsilon term.
            if summary_simulated_delta > epsilon:
                states = states + [[[theta_anc_proposed, theta_afr_proposed, theta_afr_proposed],
                                    1, summary_simulated_delta, j]]

            # Reject our proposal. We keep our current state and increment our waiting times.
            else:
                states[-1][1] += 1

    def log_states(self, cur_j: Cursor):
        """ TODO:

        :param cur_j:
        :return:
        """

