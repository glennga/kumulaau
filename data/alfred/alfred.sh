#!/bin/bash

# Iterate through all TSV files in the data/alfred folder.
for f in ${MICRO_SAT_PATH}/data/alfred/*.tsv; do
    echo "File: $f"
    python3 ${MICRO_SAT_PATH}/data/alfred/alfred.py $f
done
