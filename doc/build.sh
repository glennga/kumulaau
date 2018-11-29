#!/bin/bash

# Ensure that we have only 1 argument passed.
if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: build.sh [file to build] [data-folder]"
    exit 1
fi

# Output the 'c' and 'd' values to data files.
if [ "$#" -eq 2 ]; then
    # Save all values of 'c' to c.dat.
    echo "c" > ${MICRO_SAT_PATH}/doc/c.dat
    sqlite3 $2/all-methoda.db "SELECT C \
                               FROM WAIT_MODEL" \ |
        tail -n +3 >> ${MICRO_SAT_PATH}/doc/c.dat

    # Repeat for 'd' to d.dat.
    echo "d" > ${MICRO_SAT_PATH}/doc/d.dat
    sqlite3 $2/all-methoda.db "SELECT D \
                               FROM WAIT_MODEL" \ |
        tail -n +3 >> ${MICRO_SAT_PATH}/doc/d.dat

    # Get the traces from all runs.
    for i in 1 2 3; do
        echo "c d p" > ${MICRO_SAT_PATH}/doc/trace${i}.dat
        sleep 0.1
        sqlite3 $2/methoda-${i}.db -separator " " "SELECT C, D, PROPOSED_TIME \
                                                   FROM WAIT_MODEL \
                                                   WHERE rowid % 3 = 0 AND \
                                                   PROPOSED_TIME < 55000" \ |
            tail -n +3 >> ${MICRO_SAT_PATH}/doc/trace${i}.dat
    done
fi

# Build everything in the doc directory.
cd ${MICRO_SAT_PATH}/doc
pdflatex -shell-escape $1; sleep 0.1
bibtex $(expr $1 : "\(.*\).tex").aux; sleep 0.1
pdflatex -shell-escape $1; sleep 0.1
pdflatex -shell-escape $1; sleep 0.1

# Move everything but the LaTeX files and the PDF to some directory in build.
mkdir ${MICRO_SAT_PATH}/doc/build ${MICRO_SAT_PATH}/doc/pdf 2>/dev/null
mv *.aux *.bbl *.blg *.log *.out *.nav *.snm *.toc *.dat *-figure*.pdf *.dpth *.md5 *.dep *.auxlock \
    ${MICRO_SAT_PATH}/doc/build 2>/dev/null
mv *.pdf ${MICRO_SAT_PATH}/doc/pdf 2>/dev/null
