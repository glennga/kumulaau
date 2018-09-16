#!/usr/bin/env python3
from single import ModelParameters
from sqlite3 import Cursor
from typing import List, Dict


def create_table(cur_j: Cursor) -> None:
    """ Create the tables to log the results of some population model MCMC to. Our data model is normalized to
    accommodate different population migration models. Each model varies in which real samples are assigned to each
    population, and what parameters exist in the first place.

    :param cur_j: Cursor to the database file to log to.
    :return: None.
    """
    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_REAL (
            TIME_R TIMESTAMP,
            REAL_SAMPLE_UID TEXT,
            REAL_LOCUS TEXT,
            POP TEXT
        );""")

    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_N (
            TIME_R TIMESTAMP,
            ACCEPTANCE_TIME INT,
            POP TEXT,
            N_TYPE TEXT, 
            N_VALUE INT
        );""")

    cur_j.execute("""
        CREATE TABLE IF NOT EXISTS WAIT_MIGRATION (
            TIME_R TIMESTAMP,
            MU FLOAT,
            S FLOAT,
            KAPPA INT,
            OMEGA INT,
            U FLOAT,
            V FLOAT,
            M FLOAT,
            P FLOAT,
            WAITING INT,
            DELTA FLOAT,
            ACCEPTANCE_TIME INT
        );""")


def log_states(cur_j: Cursor,

               rsu: List[str], l: List[str], chain: List) -> None:
    """ Record our states to the appropriate table in some database.

    :param cur_j: Cursor to the database file to log to.
    :param rsu: IDs of the real sample data sets to compare to.
    :param l: Loci of the real samples to compare to.
    :param chain: States and associated times & probabilities collected after running MCMC.
    :return: None.
    """
    from datetime import datetime
    d_t = datetime.now()

    # # Record our real sample log strings and datetime.
    # cur_j.executemany("""
    #     INSERT INTO WAIT_REAL
    #     VALUES (?, ?, ?)
    # """, ((d_t, a[0], a[1]) for a in zip(rsu, l)))
    # 
    # cur_j.executemany(f"""
    #     INSERT INTO WAIT_{name}
    #     VALUES ({','.join('?' for _ in range(len(chain[0][0]) + 5))});
    # """, ((d_t, ) + tuple(a[0]) + (a[1], a[2], a[3]) for a in chain))

class AVNAParameters(ModelParameters):
    def __init__(self, i_0: ndarray, mu: float, s: float, kappa: int, omega: int, u: float, v: float, m: float,
                 p: float, big_n_anc: int, branch_n_anc: int, big_n_afr: int, big_n_naf: int):
        """

        :param i_0:
        :param mu:
        :param s:
        :param kappa:
        :param omega:
        :param u:
        :param v:
        :param m:
        :param p:
        :param big_n_anc:
        :param branch_n_anc:
        :param big_n_afr:
        :param big_n_naf:
        """
        super(AVNAParameters, self).__init__()




class AVNA:
    def __init__(self, j: Dict, rfs_d: List, cur_j: Cursor):
        """ Constructor. Load our parameters from the dictionary (from JSON), and ensure that all exist here.
        
        :param j: Dictionary of configuration parameters describing AVNA.
        """
        j_s, j_r = j['simulation'], j['real-samples']

        self.theta_0 = {
            k: ModelParameters(i_0=choose_i_0(rfs_d), big_n=j_s[k]['big_n'], mu=j_s[k]['mu'], s=j_s[k]['s'],
                               kappa=j_s[k]['kappa'], omega=j_s[k]['omega'], u=j_s[k]['u'], v=j_s[k]['v'],
                               m=j_s[k]['m'], p=j_s[k]['p'], ) for k in ['anc', 'afr', 'naf']
        }
        self.theta_0_sigma = {
            k: ModelParameters(i_0=choose_i_0(rfs_d), big_n=j_s[k]['sigma']['big_n'], mu=j_s[k]['sigma']['mu'],
                               s=j_s[k]['sigma']['s'], kappa=j_s[k]['sigma']['kappa'], omega=j_s[k]['sigma']['omega'],
                               u=j_s[k]['sigma']['u'], v=j_s[k]['sigma']['v'], m=j_s[k]['sigma']['m'],
                               p=j_s[k]['sigma']['p']) for k in ['anc', 'afr', 'naf']
        }

        self.rfs_d = {
            k: list(map(lambda a, b: cur_j.execute(""" -- Pull the frequency distributions from the real database. --
                SELECT ELL, ELL_FREQ
                FROM REAL_ELL
                WHERE SAMPLE_UID LIKE ?
                AND LOCUS LIKE ?
            """, (a, b, )).fetchall(), j_r[k]['rsu'], j_r[k]['l'])) for k in ['afr', 'naf']
        }

def avna(it: int, rs: int, rp: int, epsilon: float, j: Dict) -> List:
    """ My interpretation of an MCMC-ABC rejection sampling approach to approximate the posterior distribution of the
    AVNA (Africa vs. Non Africa) population migration model. The steps taken are as follows:

    1) We start with some initial guess and simulate individuals with an effective population size of
       set branch size (i.e. we stop at the branch size). We denote this as population ANC.
    2) The individuals of ANC are them randomly used as seeds for two child populations: AFR (Africa) and NAF
       (Non-Africa). Simulate these populations in serial (parallelism involved with simulation one population is well
       used, I think).
    3) Compute the average distance between a set of simulated AFR and real AFR (delta). Repeat for NAF.
        a) If this term is above some defined epsilon, we append this to our chain.
        b) Otherwise, we reject it and increase the waiting time of our current parameter state by one.
    3) Repeat for 'it' iterations.

    Source:
    https://theoreticalecology.wordpress.com/2012/07/15/a-simple-approximate-bayesian-computation-mcmc-abc-mcmc-in-r/
    
    :param it: Number of iterations to run MCMC for.
    :param rs: Number of samples per simulation to use to obtain delta.
    :param rp: Number of simulations to use to obtain delta.
    :param epsilon: Minimum acceptance value for delta.
    :param j: Configuration parameters, loaded from a JSON file.
    :return: A chain of all states we visited (parameters), their associated waiting times, and the sum total of their
             acceptance probabilities.
    """
    
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    from model import choose_i_0
    from sqlite3 import connect
    from json import loads

    parser = ArgumentParser(description='ABC MCMC for the *human migration model* parameter estimation.')
    parser.add_argument('-rdb', help='Location of the real database file.', type=str, default='data/real.db')
    parser.add_argument('-edb', help='Location of the database to record to.', type=str, default='data/migration.db')
    parser.add_argument('-mig', help='Type of migration model to run.', type=str, choices=['AVNA'])
    parser.add_argument('-json', help='Location of the configuration file attached to the migration model.', type=str)
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    paa('-rp', 'Number of simulations to use to obtain delta.', int)
    paa('-rs', 'Number of samples per simulation to use to obtain delta.', int)
    paa('-epsilon', 'Minimum acceptance value for delta (ABC only).', float)
    paa('-it', 'Number of iterations to run MCMC for.', int)

    c_args = parser.parse_args()  # Parse our arguments.
    j_args = loads(c_args.json)

    # Connect to all of our databases.
    conn_r, conn_e = connect(c_args.rdb), connect(c_args.edb)
    cur_r, cur_e = conn_r.cursor(), conn_e.cursor()

    freq_r = cur_r.execute(""" -- Pull the frequency distributions from the real database. --
        SELECT ELL, ELL_FREQ
        FROM REAL_ELL
    """).fetchall()

    two_nm = list(map(lambda a, b: int(cur_r.execute(""" -- Retrieve the sample sizes, the number of alleles. --
        SELECT SAMPLE_SIZE
        FROM REAL_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, (a, b,)).fetchone()[0]), c_args.rsu, c_args.l))

    # Construct our parameter set.
    theta_0_m = ModelParameters(i_0=choose_i_0(freq_r), big_n=c_args.big_n, mu=c_args.mu, s=c_args.s, kappa=c_args.kappa,
                                omega=c_args.omega, u=c_args.u, v=c_args.v, m=c_args.m, p=c_args.p)
    theta_s_m = ModelParameters(i_0=choose_i_0(freq_r), big_n=c_args.big_n_sigma, mu=c_args.mu_sigma, s=c_args.s_sigma,
                                kappa=c_args.kappa_sigma, omega=c_args.omega_sigma, u=c_args.u_sigma, v=c_args.v_sigma,
                                m=c_args.m_sigma, p=c_args.p_sigma)

    # Determine which population model we are simulating.
    if c_args.mig.casefold == 'avna':
        create_table(cur_e, 'AVNA', ['BRANCH_N_ANC', 'BIG_N_AFR', 'BIG_N_NAF'])
        theta_0_m.child_n, theta_s_m = [c_args.branch_n_anc, c_args.big_n_afr, c_args.big_n_naf], \
                                       [c_args.branch_n_anc_sigma, c_args.big_n_afr_sigma, c_args.big_n_naf_sigma]

    # Perform the MCMC, and record our chain.
    log_states(cur_e, 'AVNA', c_args.rsu, c_args.l, avna(c_args.it, c_args.rs, c_args.rp, two_nm, c_args.epsilon,
                                                         theta_0_m, theta_s_m))
    conn_e.commit(), conn_r.close(), conn_e.close()



