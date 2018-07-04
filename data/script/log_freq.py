# coding=utf-8
""" Record all of the frequency data obtained from ALFRED to a SQLite database. This will be used to estimate the
parameters involved in our simulation models. We assume the following schema for the tab-separated files:

    popName	popUId	sampleUId	2N	locusSymbol	siteName	alleleSymbol	entryDate	frequency

"""


def create_table() -> None:
    """ Create the required table if it does not exist.

    :return: None.
    """
    cur.execute("""
        CREATE TABLE IF NOT EXISTS REAL_ELL (
            TIME_R TIMESTAMP,
            POP_NAME TEXT,
            POP_UID TEXT,
            SAMPLE_UID TEXT,
            SAMPLE_SIZE INT,
            LOCUS TEXT,
            ELL TEXT,
            ELL_FREQ FLOAT
        );""")


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sqlite3 import connect
    from csv import reader

    # We grab our arguments.
    parser = ArgumentParser(description='Record frequency data from ALFRED in TSV format to a database.')
    parser.add_argument('freq_f', help='Frequency file in TSV format')
    parser.add_argument('-f', '--log', help='The location of the database to log to.', default='data/real.db')
    args = parser.parse_args()

    # Connect to the database to log to.
    conn = connect(args.log)
    cur = conn.cursor()
    create_table()

    # Open the TSV file. Skip the header. Read the rest of the entries.
    with open(args.freq_f) as tsv_f:
        freq_reader = reader(tsv_f, delimiter='\t')
        next(freq_reader)

        # Perform the insertion. Print out any anomalies.
        for entry in freq_reader:
            try:
                cur.execute("""
                    INSERT INTO REAL_ELL (TIME_R, POP_NAME, POP_UID, SAMPLE_UID, SAMPLE_SIZE, LOCUS, ELL, ELL_FREQ)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (entry[7], entry[0], entry[1], entry[2], int(entry[3]), entry[4], int(entry[6]), float(entry[8])))
            except (ValueError, IndexError):
                print('Error at: {}'.format(entry))

    conn.commit(), conn.close()

