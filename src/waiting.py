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


def histogram_1(cursor: Cursor, step_sizes: Dict[str, float]) -> None:
    """ Display the results of the MCMC script for our mutation model: a histogram of waiting times for each parameter.

    :param cursor: Cursor to the MCMC database to pull the data from.
    :param step_sizes: Histogram step sizes for the N, F, C, U, D, KAPPA, OMEGA dimensions.
    :return: None.
    """
    from itertools import chain
    from numpy import arange

    set_style(), plt.suptitle(r'Waiting Times for Mutation Model MCMC')

    dimension_labels = [r'$N$', r'$f$', r'$c$', r'$u$', r'$d$', r'$\kappa$', r'$\Omega$']
    for j, dimension in enumerate(['N', 'F', 'C', 'U', 'D', 'KAPPA', 'OMEGA']):
        plt.subplot(2, 4, j + 1)

        # Grab the minimum and maximum values for the current dimension.
        min_dimension, max_dimension = [float(x) for x in cursor.execute(f"""
          SELECT MIN(CAST({dimension} AS FLOAT)), MAX(CAST({dimension} AS FLOAT))
          FROM WAIT_MODEL
        """).fetchone()]

        if min_dimension != max_dimension:  # Our bin widths depend on the current dimension.
            bins = arange(min_dimension, max_dimension,
                          (step_sizes[dimension] if dimension not in ['BIG_N', 'KAPPA', 'OMEGA'] else 1))
        else:
            bins = 'auto'

        # Obtain the data to plot. Waiting times indicate the number of repeats to apply to this state.
        axis_wait = cursor.execute(f"""
          SELECT CAST({dimension} AS FLOAT), WAITING_TIME
          FROM WAIT_MODEL
        """).fetchall()
        axis = list(chain.from_iterable([[float(a[0]) for _ in range(int(a[1]))] for a in axis_wait]))

        # Plot the histogram, and find the best fit line (assuming beta distribution).
        plt.gca().set_title(dimension_labels[j]), plt.hist(axis, bins=bins, density=True, histtype='stepfilled')

    plt.subplots_adjust(top=0.909, bottom=0.051, left=0.032, right=0.983, hspace=0.432, wspace=0.135)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect
    from matplotlib import use

    # Generate our plot help strings.
    plot_help = """ Visualization function to use: 
        [1 <- Waiting times histogram of mutation model MCMC.]
    """
    plot_parmeters_help = """ Parameters associated with function of use:
        [1 <- Step sizes of histogram in following order: N, F, C, U, D, KAPPA, OMEGA.] 
    """

    parser = ArgumentParser(description='Display the results of MCMC scripts.')
    parser.add_argument('-db', help='Location of the database required to operate on.', type=str)
    parser.add_argument('-plot', help=plot_help, type=int, choices=[1])
    parser.add_argument('-image_file', help='Image file to save resulting figure to.', type=str)
    parser.add_argument('plot_parameters', help='Parameters associated with function of use.', type=float, nargs='+')
    main_arguments = parser.parse_args()  # Parse our arguments.

    # Connect to the appropriate database.
    connection_r = connect(main_arguments.db)
    cursor_r = connection_r.cursor()

    # Determine if we need to use X-server (plotting to display).
    use('Agg') if main_arguments.image_file is not None else None
    from matplotlib import pyplot as plt

    # Perform the appropriate function.
    if main_arguments.plot == 1 and len(main_arguments.plot_parameters) == 7:
        histogram_1(cursor_r, dict(zip(['N', 'F', 'C', 'U', 'D', 'KAPPA', 'OMEGA'], main_arguments.plot_parameters)))
    else:
        print('Incorrect number of plot parameters.') and exit(-1)

    # Save or display, your choice!
    plt.savefig(main_arguments.image_file, dpi=100) if main_arguments.image_file is not None else plt.show()
