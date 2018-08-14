from sqlite3 import Cursor
from typing import Dict


def emcee(cur_j: Cursor, b: int, hs: Dict[str, float]) -> None:
    """ Display the results of the MCMC script: a histogram of each parameter.

    :param cur_j: Cursor to the MCMC database to pull the data from.
    :param b: Burn in period. Number of trials to remove before displaying results.
    :param hs: Histogram step sizes for MU, S, U, V, M, and P dimensions.
    :return: None.
    """
    from matplotlib import pyplot as plt
    from numpy import arange
    plt.suptitle('Posterior Distribution Samples (MCMC)')

    # Determine how many rows exist, required to remove burn in period.
    cardinality = int(cur_j.execute("""
      SELECT MAX(ROWID)
      FROM WAIT_POP
    """).fetchone()[0])

    for j, dimension in enumerate(['BIG_N', 'MU', 'S', 'KAPPA', 'OMEGA', 'U', 'V', 'M', 'P']):
        plt.subplot(3, 3, j + 1)

        # Grab the mininum and maximum values for the current dimension.
        min_dimension, max_dimension = [float(x) for x in cur_j.execute(f"""
          SELECT MIN(CAST({dimension} AS FLOAT)), MAX(CAST({dimension} AS FLOAT))
          FROM WAIT_POP
        """).fetchone()]

        # Our bin widths depend on the current dimension.
        if min_dimension != max_dimension:
            bins = arange(min_dimension, max_dimension, (hs[dimension] if dimension not in
                                                         ['BIG_N', 'KAPPA', 'OMEGA'] else 1))
        else:
            bins = 'auto'

        # Plot the histogram.
        plt.gca().set_title(dimension)
        plt.hist([float(x[0]) for x in cur_j.execute(f"""
          SELECT CAST({dimension} AS FLOAT) -- Casting required for I_0. --
          FROM WAIT_POP
          ORDER BY ROWID DESC
          LIMIT {cardinality - b}
        """).fetchall()], bins=bins)

    plt.subplots_adjust(top=0.924, bottom=0.051, left=0.032, right=0.983, hspace=0.432, wspace=0.135)
    plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect

    parser = ArgumentParser(description='Display the results of various scripts (emcee, ...).')
    parser.add_argument('-db', help='Location of the database required to operate on.', type=str)
    parser.add_argument('-f', help='Data to display.', type=str, choices=['emcee'])
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    paa('-b', '(emcee) Burn in period. Number of trials to remove before displaying results.', int)
    parser.add_argument('-hs', help='(emcee) Histogram step sizes in order: MU, S, U, V, M, P', nargs=6, type=float)
    args = parser.parse_args()  # Parse our arguments.

    # Connect to the appropriate database.
    conn = connect(args.db)
    cur = conn.cursor()

    # Perform the appropriate function.
    if args.f == 'emcee':
        emcee(cur, args.b, dict(zip(['MU', 'S', 'U', 'V', 'M', 'P'], args.hs)))
