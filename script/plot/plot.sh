#!/bin/bash

# Ensure that we only have 1 argument passed.
if [[ "$#" -ne 1 ]]; then
    echo "Usage: plot.sh [location of database files]"
    exit 1
fi

# Parameters for all functions.
BURN_IN=15000
PARAMS="0 0 0.001 0.0001 0 0"

# Visualize all instances of our mutation model MCMC.
for f in $1/*.db; do
    echo "Plotting [${f}, Waiting Times]."
    python3 script/plot/plot.py \
        -db ${f} \
        -function 1 \
        -image ${f%".db"}-waiting.png \
        -burn_in ${BURN_IN} \
        -params $(echo "$PARAMS")

    echo "Plotting [${f}, Frequency]."
    python3 script/plot/plot.py \
        -db ${f} \
        -function 2 \
        -image ${f%".db"}-frequency.png \
        -burn_in ${BURN_IN} \
        -params $(echo "$PARAMS")

    echo "Plotting [${f}, Trace]."
    python3 script/plot/plot.py \
        -db ${f} \
        -function 3 \
        -burn_in ${BURN_IN} \
        -image ${f%".db"}-trace.png

    # Takes too long right now...
#    echo "Plotting [${f}, Likelihood]."
#    python3 script/plot/plot.py \
#        -db ${f} \
#        -function 4 \
#        -burn_in ${BURN_IN} \
#        -image ${f%".db"}-likelihood.png
done

# Combine all samples of our mutation model MCMC.
cp $(find $1 -maxdepth 1 -name methoda-*.db | head -1) $1/all.db
sqlite3 $1/all.db "DELETE FROM WAIT_MODEL \
                   WHERE 1 = 1;"

# Iterate through each of the generated databases, and copy this table to the head. This must be sequential.
for f in $1/*.db; do
	sqlite3 $1/all.db "ATTACH '${f}' AS A; \
	                   INSERT INTO WAIT_MODEL \
	                   SELECT * \
	                   FROM A.WAIT_MODEL; \
	                   DETACH DATABASE A;"
done

# Plot a frequency graph for all.
echo "Plotting [All, Frequency]."
python3 src/plot.py \
    -db $1/all.db \
    -function 2 \
    -image data/all-methoda-frequency.png \
    -burn_in ${BURN_IN} \
    -params $(echo "$PARAMS")
