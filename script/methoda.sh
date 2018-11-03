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

python3 src/methoda.py \
    -mdb "data/methoda-$1.db" \
    -simulation_n 100 \
    -epsilon 0.60 \
    -iterations_n 20000 \
    -uid_observed ${sample_uids} \
    -locus_observed ${loci} \
    -n 100 -n_sigma 0.0 \
    -f 1 -f_sigma 0.0 \
    -c 0.01067 -c_sigma 0.00503 \
    -u 1.23414 -u_sigma 0.01062 \
    -d 0.00255 -d_sigma 0.00068 \
    -kappa 3 -kappa_sigma 0.0 \
    -omega 30 -omega_sigma 0.0

echo "MCMC is finished!"
