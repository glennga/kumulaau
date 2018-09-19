from sqlite3 import Cursor
from typing import Dict


def set_style() -> None:
    """ Change the default matplotlib settings.

    :return: None.
    """
    from matplotlib import rc

    plt.style.use('bmh')  # Change that ugly default matplotlib look.
    rc('text', usetex=True), plt.rc('font', family='serif')  # Use TeX.

    plt.figure(figsize=(13, 7), dpi=100)


def model(cur_j: Cursor, hs: Dict[str, float]) -> None:
    """ Display the results of the MCMC script for our mutation model: a histogram of each parameter.

    :param cur_j: Cursor to the MCMC database to pull the data from.
    :param hs: Histogram step sizes for MU, S, U, V, M, and P dimensions.
    :return: None.
    """
    from scipy.stats import beta
    from numpy import arange, linspace
    from itertools import chain

    set_style(), plt.suptitle(r'Posterior Distribution Samples (MCMC)')

    dimension_labels = [r'$N$', r'$\mu$', r'$s$', r'$K$', r'$\Omega$', r'$u$', r'$v$', r'$m$', r'$p$']
    for j, dimension in enumerate(['BIG_N', 'MU', 'S', 'KAPPA', 'OMEGA', 'U', 'V', 'M', 'P']):
        plt.subplot(3, 3, j + 1)

        # Grab the minimum and maximum values for the current dimension.
        min_dimension, max_dimension = [float(x) for x in cur_j.execute(f"""
          SELECT MIN(CAST({dimension} AS FLOAT)), MAX(CAST({dimension} AS FLOAT))
          FROM WAIT_MODEL
        """).fetchone()]

        # Our bin widths depend on the current dimension.
        if min_dimension != max_dimension:
            bins = arange(min_dimension, max_dimension, (hs[dimension] if dimension not in
                                                         ['BIG_N', 'KAPPA', 'OMEGA'] else 1))
        else:
            bins = 'auto'

        # Obtain the data to plot. Waiting times indicate the number of repeats to apply to this state.
        axis_wait = cur_j.execute(f"""
          SELECT CAST({dimension} AS FLOAT), WAITING
          FROM WAIT_MODEL
        """).fetchall()
        axis = list(chain.from_iterable([[float(a[0]) for _ in range(int(a[1]))] for a in axis_wait]))

        # Plot the histogram, and find the best fit line (assuming beta distribution).
        plt.gca().set_title(dimension_labels[j]), plt.hist(axis, bins=bins, density=True, histtype='stepfilled')
        if axis.count(axis[0]) != len(axis):  # Do not plot for constant dimensions.
            spc, c_ylim = linspace(min(plt.xticks()[0]), max(plt.xticks()[0]), len(axis)), plt.ylim()
            ab, bb, cb, db = beta.fit(axis)
            plt.plot(spc, beta.pdf(spc, ab, bb, cb, db)), plt.ylim(c_ylim)  # Retain past y-limits, focus is histogram.

    plt.subplots_adjust(top=0.924, bottom=0.051, left=0.032, right=0.983, hspace=0.432, wspace=0.135)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect
    from matplotlib import use

    parser = ArgumentParser(description='Display the results of various scripts (posterior, ...).')
    parser.add_argument('-db', help='Location of the database required to operate on.', type=str)
    parser.add_argument('-f', help='Data to display.', type=str, choices=['model'])
    paa = lambda paa_1, paa_2, paa_3: parser.add_argument(paa_1, help=paa_2, type=paa_3)

    paa('-image', 'Image file to save resulting repeat length distribution (histogram) to.', str)
    parser.add_argument('-hs', help='(posterior) Histogram step sizes in order: MU, S, U, V, M, P', nargs=6, type=float)
    args = parser.parse_args()  # Parse our arguments.

    # Connect to the appropriate database.
    conn = connect(args.db)
    cur = conn.cursor()

    # Determine if we need to use X-server (plotting to display).
    use('Agg') if args.image is not None else None
    from matplotlib import pyplot as plt

    # Perform the appropriate function.
    if args.f == 'model':
        model(cur, dict(zip(['MU', 'S', 'U', 'V', 'M', 'P'], args.hs)))

    # Save or display, your choice!
    plt.savefig(args.image, dpi=100) if args.image is not None else plt.show()
