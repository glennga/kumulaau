from sqlite3 import Cursor
from typing import Dict


def set_style() -> None:
    """ Change the default matplotlib settings.

    :return: None.
    """
    from matplotlib import rc

    # Change that ugly default matplotlib look and use TeX.
    plt.style.use('bmh')
    rc('text', usetex=True), plt.rc('font', family='serif')
    plt.figure(figsize=(13, 7), dpi=100)


def histogram_1(cursor: Cursor, step_sizes: Dict[str, float], burn_in: int) -> None:
    """ Display the results of the MCMC script for our mutation model: a histogram of waiting times for each parameter.

    :param cursor: Cursor to the MCMC database to pull the data from.
    :param step_sizes: Histogram step sizes for the N, F, C, U, D, KAPPA, OMEGA dimensions.
    :param burn_in: Burn in period. We only use datum whose acceptance time falls after this.
    :return: None.
    """
    from itertools import chain
    from numpy import arange

    set_style(), plt.suptitle(r'Parameter Waiting Times (Frequency) for Mutation Model MCMC')

    dimension_labels = [r'$N$', r'$f$', r'$c$', r'$d$', r'$\kappa$', r'$\Omega$']
    for j, dimension in enumerate(['N', 'F', 'C', 'D', 'KAPPA', 'OMEGA']):
        plt.subplot(2, 3, j + 1)

        # Grab the minimum and maximum values for the current dimension.
        min_dimension, max_dimension = [float(x) for x in cursor.execute(f"""
            SELECT MIN(CAST({dimension} AS FLOAT)), MAX(CAST({dimension} AS FLOAT))
            FROM WASTEFUL_MODEL
            WHERE PROPOSED_TIME > {burn_in}
        """).fetchone()]

        if min_dimension != max_dimension:  # Our bin widths depend on the current dimension.
            bins = arange(min_dimension, max_dimension,
                          (step_sizes[dimension] if dimension not in ['BIG_N', 'KAPPA', 'OMEGA'] else 1))
        else:
            bins = 'auto'

        # Obtain the data to plot. Waiting times indicate the number of repeats to apply to this state.
        axis_wait = cursor.execute(f"""
            SELECT CAST({dimension} AS FLOAT), WAITING_TIME
            FROM WASTEFUL_MODEL
            WHERE PROPOSED_TIME > {burn_in}
        """).fetchall()
        axis = list(chain.from_iterable([[float(a[0]) for _ in range(int(a[1]))] for a in axis_wait]))

        # Plot the histogram, and find the best fit line (assuming beta distribution).
        plt.gca().set_title(dimension_labels[j]), plt.hist(axis, bins=bins, density=True, histtype='stepfilled')

    plt.subplots_adjust(top=0.909, bottom=0.051, left=0.032, right=0.983, hspace=0.432, wspace=0.135)


def histogram_2(cursor: Cursor, step_sizes: Dict[str, float], burn_in: int) -> None:
    """ Display the results of the MCMC script for our mutation model: a histogram of posterior samples for each
     parameter and an overlaid beta distribution estimation.

    :param cursor: Cursor to the MCMC database to pull the data from.
    :param step_sizes: Histogram step sizes for the N, F, C, U, D, KAPPA, OMEGA dimensions.
    :param burn_in: Burn in period. We only use datum whose acceptance time falls after this.
    :return: None.
    """
    from scipy.stats import beta, norm, gamma
    from numpy import arange, linspace

    set_style(), plt.suptitle(r'Parameter Frequency for Mutation Model MCMC')

    dimension_labels = [r'$N$', r'$f$', r'$c$', r'$d$', r'$\kappa$', r'$\Omega$']
    for j, dimension in enumerate(['N', 'F', 'C', 'D', 'KAPPA', 'OMEGA']):
        plt.subplot(2, 3, j + 1)

        # Grab the minimum and maximum values for the current dimension.
        min_dimension, max_dimension = [float(x) for x in cursor.execute(f"""
            SELECT MIN(CAST({dimension} AS FLOAT)), MAX(CAST({dimension} AS FLOAT))
            FROM WASTEFUL_MODEL
            WHERE PROPOSED_TIME > {burn_in}
        """).fetchone()]

        if min_dimension != max_dimension:  # Our bin widths depend on the current dimension.
            bins = arange(min_dimension, max_dimension,
                          (step_sizes[dimension] if dimension not in ['BIG_N', 'KAPPA', 'OMEGA'] else 1))
        else:
            bins = 'auto'

        # Obtain the data to plot.
        axis = list(map(lambda a: float(a[0]), cursor.execute(f"""
            SELECT CAST({dimension} AS FLOAT)
            FROM WASTEFUL_MODEL
            WHERE PROPOSED_TIME > {burn_in}
        """).fetchall()))

        # Plot the histogram, and find the best fit line.
        plt.gca().set_title(dimension_labels[j]), plt.hist(axis, bins=bins, density=True, histtype='stepfilled')
        if axis.count(axis[0]) != len(axis):
            spc, c_ylim = linspace(min(plt.xticks()[0]), max(plt.xticks()[0]), len(axis)), plt.ylim()

            ab, bb, cb, db = beta.fit(axis)  # Beta distribution plot (red).
            plt.plot(spc, beta.pdf(spc, ab, bb, cb, db), '#A60628')
            print(f'\tBeta [{dimension}]: {ab}, {bb}, {cb}, {db}')

            an, bn = norm.fit(axis)  # Gaussian distribution plot (purple).
            plt.plot(spc, norm.pdf(spc, an, bn), '#7A68A6')
            print(f'\tGaussian [{dimension}]: {an}, {bn}')

            ag, bg, cg = gamma.fit(axis)  # Gamma distribution plot (green).
            plt.plot(spc, gamma.pdf(spc, ag, bg, cg), '#467821')
            print(f'\tGamma [{dimension}]: {ag}, {bg}, {cg}')

            plt.ylim(c_ylim)  # Focus is our histogram, not our line of fits.

    plt.subplots_adjust(top=0.909, bottom=0.051, left=0.032, right=0.983, hspace=0.432, wspace=0.135)


def trace_1(cursor: Cursor, burn_in: int) -> None:
    """ TODO:

    :param cursor: Cursor to the MCMC database to pull the data from.
    :param burn_in: Burn in period. We only use datum whose acceptance time falls after this.
    :return: None.
    """
    set_style(), plt.suptitle(r'Trace Plot for Mutation Model MCMC')

    dimension_labels = [r'$N$', r'$f$', r'$c$', r'$d$', r'$\kappa$', r'$\Omega$']
    for j, dimension in enumerate(['N', 'F', 'C', 'D', 'KAPPA', 'OMEGA']):
        plt.subplot(2, 3, j + 1)

        # Obtain the data to plot. The acceptance time indicates how we sort our data.
        axis_accept = list(map(lambda a: [int(a[0]), float(a[1])], cursor.execute(f"""
            SELECT PROPOSED_TIME, CAST({dimension} AS FLOAT)
            FROM WASTEFUL_MODEL
            WHERE PROPOSED_TIME > {burn_in}
            ORDER BY PROPOSED_TIME
        """).fetchall()))
        x_axis, y_axis = list(zip(*axis_accept))

        # Plot each point as (proposed_time, dimension).
        plt.gca().set_title(dimension_labels[j]), plt.plot(x_axis, y_axis)

    plt.subplots_adjust(top=0.909, bottom=0.051, left=0.032, right=0.983, hspace=0.432, wspace=0.135)


def likelihood_1(cursor: Cursor, burn_in: int) -> None:
    """ TODO:

    :param cursor: Cursor to the MCMC database to pull the data from.
    :param burn_in: Burn in period. We only use datum whose acceptance time falls after this.
    :return:
    """
    from numpy import log

    set_style(), plt.suptitle(r'Parameter Log-Likelihood for Mutation Model MCMC')

    dimension_labels = [r'$N$', r'$f$', r'$c$', r'$d$', r'$\kappa$', r'$\Omega$']
    for j, dimension in enumerate(['N', 'F', 'C', 'D', 'KAPPA', 'OMEGA']):
        plt.subplot(2, 3, j + 1)

        # Obtain the data to plot. We only take every other value.
        axis_accept = list(map(lambda a: [float(a[0]), float(a[1])], cursor.execute(f"""
            SELECT LIKELIHOOD, CAST({dimension} AS FLOAT)
            FROM WASTEFUL_MODEL
            WHERE rowid % 2 = 0
            AND PROPOSED_TIME > {burn_in}
            ORDER BY {dimension}
        """).fetchall()))
        y_axis, x_axis = list(zip(*axis_accept))

        # Plot each point as (dimension, log likelihood).
        plt.gca().set_title(dimension_labels[j]), plt.bar(x_axis, log(y_axis))

    plt.subplots_adjust(top=0.909, bottom=0.051, left=0.032, right=0.983, hspace=0.432, wspace=0.135)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect
    from matplotlib import use

    # Generate our plot help strings.
    plot_help = """ Visualization function to use:
        [1 <- Waiting times histogram of mutation model MCMC.]
        [2 <- Probability of our mutation model parameters given our data (histogram & MCMC).]
        [3 <- Trace plot of our parameters for the mutation model MCMC.]
        [4 <- Log-likelihood curves of our parameters for the mutation model MCMC.]
    """
    plot_parmeters_help = """ Parameters associated with function of use:
        [1 <- Step sizes of histogram in following order: N, F, C, D, KAPPA, OMEGA.]         
        [2 <- Step sizes of histogram in following order: N, F, C, D, KAPPA, OMEGA.] 
        [3 <- None.]
        [4 <- None.]
    """

    parser = ArgumentParser(description='Display the results of MCMC scripts.')
    parser.add_argument('-db', help='Location of the database required to operate on.', type=str)
    parser.add_argument('-burn_in', help='Burn in period, in terms of iterations.', type=int)
    parser.add_argument('-function', help=plot_help, type=int, choices=[1, 2, 3, 4])
    parser.add_argument('-image_file', help='Image file to save resulting figure to.', type=str)
    parser.add_argument('-params', help=plot_parmeters_help, type=float, nargs='+')
    main_arguments = parser.parse_args()  # Parse our arguments.

    # Connect to the appropriate database.
    connection_r = connect(main_arguments.db)
    cursor_r = connection_r.cursor()

    # Determine if we need to use X-server (plotting to display).
    use('Agg') if main_arguments.image_file is not None else use('TkAgg')
    from matplotlib import pyplot as plt

    # Perform the appropriate function. Do you like this makeshift switch statement? :-)
    dict_arg = lambda: dict(zip(['N', 'F', 'C', 'D', 'KAPPA', 'OMEGA'], main_arguments.params))
    {
        1: lambda: histogram_1(cursor_r, dict_arg(), main_arguments.burn_in),
        2: lambda: histogram_2(cursor_r, dict_arg(), main_arguments.burn_in),
        3: lambda: trace_1(cursor_r, main_arguments.burn_in),
        4: lambda: likelihood_1(cursor_r, main_arguments.burn_in)
    }.get(main_arguments.function)()

    # Save or display, your choice!
    plt.savefig(main_arguments.image_file, dpi=100) if main_arguments.image_file is not None else plt.show()
