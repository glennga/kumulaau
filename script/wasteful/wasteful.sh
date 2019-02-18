#!/bin/bash

# Our criteria for the loci and sample IDs to use for this run of MCMC: The Colombian populace.
sample_uids=(); loci=()
for r in $(sqlite3 data/observed.db "SELECT DISTINCT SAMPLE_UID, LOCUS \
                                     FROM OBSERVED_ELL \
                                     WHERE LOCUS LIKE 'D16S539' \
                                     AND POP_UID LIKE 'PO000503I';"); do
    IFS='|' read -r -a array <<< "$r"
    sample_uids+="${array[0]} "; loci+="${array[1]} "
done

# Run once to seed our database. Must break into parts because GC is garbage ):<
python3 src/methoda.py \
    -mdb "data/$1.db" \
    -simulation_n 100 \
    -epsilon 0.55 \
    -iterations_n 10000 \
    -seed 0 \
    -flush_n 5000 \
    -uid_observed ${sample_uids} \
    -locus_observed ${loci} \
    -n 50 -n_sigma 0.0 \
    -f 100 -f_sigma 0.0 \
    -c 0.01067 -c_sigma 0.01003 \
    -d 0.00255 -d_sigma 0.00108 \
    -kappa 3 -kappa_sigma 0.0 \
    -omega 30 -omega_sigma 0.0
echo "MCMC Progress [1/10]."

# Repeat 9 more times.
for i in {2..10}; do
    python3 src/methoda.py \
        -mdb "data/$1.db" \
        -simulation_n 100 \
        -epsilon 0.55 \
        -iterations_n 10000 \
        -seed 1 \
        -flush_n 5000 \
        -uid_observed ${sample_uids} \
        -locus_observed ${loci} \
        -n_sigma 0.0 \
        -f_sigma 0.0 \
        -c_sigma 0.01003 \
        -d_sigma 0.00108 \
        -kappa_sigma 0.0 \
        -omega_sigma 0.0
    echo "MCMC Progress [$i/10]."
done

echo "MCMC is finished!"
