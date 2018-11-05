#!/bin/bash

# Parameters for all functions.
BURN_IN=15000
PARAMS="0 0 0.0005 0.001 0.0001 0 0"

# Visualize all instances of our mutation model MCMC.
for f in data/methoda-*.db; do
    python3 src/plot.py \
        -db ${f} \
        -function 1 \
        -image ${f%".db"}-waiting.png \
        -burn_in ${BURN_IN} \
        -params $(echo "$PARAMS")

    python3 src/plot.py \
        -db ${f} \
        -function 2 \
        -image ${f%".db"}-frequency.png \
        -burn_in ${BURN_IN} \
        -params $(echo "$PARAMS")

    python3 src/plot.py \
        -db ${f} \
        -function 3 \
        -burn_in ${BURN_IN} \
        -image ${f%".db"}-trace.png

    python3 src/plot.py \
        -db ${f} \
        -function 4 \
        -burn_in ${BURN_IN} \
        -image ${f%".db"}-likelihood.png
done

# Combine all samples of our mutation model MCMC.
cp $(find data/ -name methoda-*.db | head -1) data/all-methoda.db
sqlite3 data/all-methoda.db "DELETE FROM WAIT_MODEL \
                             WHERE 1 = 1;"

# Iterate through each of the generated databases, and copy this table to the head. This must be sequential.
for f in data/methoda-*.db; do
	sqlite3 data/all-methoda.db "ATTACH '${f}' AS A; \
	                             INSERT INTO WAIT_MODEL \
	                             SELECT * \
	                             FROM A.WAIT_MODEL \
	                             WHERE PROPOSED_TIME > ${BURN_IN}; \
	                             DETACH DATABASE A;"
done

# Plot a frequency graph for all.
python3 src/plot.py \
    -db data/all-methoda.db \
    -function 2 \
    -image data/all-methoda-frequency.png \
    -burn_in ${BURN_IN} \
    -params $(echo "$PARAMS")
