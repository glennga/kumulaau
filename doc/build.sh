#!/bin/bash

# Ensure that we have only 1 argument passed.
if [ "$#" -ne 1 ]; then
    echo "Usage: build.sh [file to build]"
    exit 1
fi

# Build everything in the doc directory.
cd ${MICRO_SAT_PATH}/doc
pdflatex $1; sleep 0.1
bibtex $(expr $1 : "\(.*\).tex").aux; sleep 0.1
pdflatex $1; sleep 0.1
pdflatex $1; sleep 0.1

# Move everything but the LaTeX files and the PDF to some directory in build.
mkdir ${MICRO_SAT_PATH}/doc/build ${MICRO_SAT_PATH}/doc/pdf 2>/dev/null
mv *.aux *.bbl *.blg *.log *.out *.nav *.snm *.toc ${MICRO_SAT_PATH}/doc/build 2>/dev/null
mv *.pdf ${MICRO_SAT_PATH}/doc/pdf 2>/dev/null
