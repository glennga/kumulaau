#!/usr/bin/env python3
from kumulaau._observed import Observed

if __name__ == '__main__':
    from argparse import ArgumentParser
    from types import SimpleNamespace
    from sqlite3 import connect
    from csv import reader

    # We assume the following schema before proceeding:
    # popName	popUId	sampleUId	2N	locusSymbol	siteName	alleleSymbol	entryDate	frequency

    # We grab our arguments.
    parser = ArgumentParser(description='Record frequency data from ALFRED in TSV format to a database.')
    parser.add_argument('freq_f', help='Frequency file in TSV format')
    parser.add_argument('-f', help='The location of the database to log to.', default='data/observed.db')
    args = parser.parse_args()

    # Connect to the database to log to.
    connection = connect(args.f)
    cursor = connection.cursor()
    Observed.create_table(cursor)

    # Open the TSV file. Skip the header. Read the rest of the entries.
    with open(args.freq_f) as tsv_f:
        freq_reader = reader(tsv_f, delimiter='\t')
        next(freq_reader)

        # Perform the insertion. Print out any anomalies.
        for entry in freq_reader:
            try:
                Observed.single_insert(cursor, (SimpleNamespace(time_r=entry[7],
                                                                pop_name=entry[0],
                                                                pop_uid=entry[1],
                                                                sample_uid=entry[2],
                                                                sample_size=int(entry[3]),
                                                                locus=entry[4],
                                                                ell=int(entry[6]),
                                                                ell_freq=float(entry[8]))))
            except (ValueError, IndexError):
                print('Error at: {}'.format(entry))
