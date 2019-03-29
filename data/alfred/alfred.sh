#!/bin/bash

# Iterate through all TSV files in the data/alfred folder.
SCRIPT_DIR=$(dirname "$0")
for f in ${SCRIPT_DIR}/*.tsv; do
    echo "File: $f"
    python3 ${SCRIPT_DIR}/alfred.py ${f} -f "${1:-data/observed.db}"
done
