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

# Run our MCMC 3 times in parallel, we compare the distributions of each 3 to verify convergence manually.
for run in 1 2 3; do
    sem -j+0 \
    python3 src/methoda.py \
        -mdb "data/methoda-${run}.db" \
        -simulation_n 10 \
        -epsilon 0.7 \
        -iterations_n 1000 \
        -uid_observed ${sample_uids} \
        -locus_observed ${loci} \
        -n 1000 -n_sigma 0.0 \
        -f 1 -f_sigma 0.0 \
        -c 0.00162 -c_sigma 0.0103 \
        -u 1.19875 -u_sigma 0.0562 \
        -d 0.00070 -d_sigma 0.0038 \
        -kappa 3 -kappa_sigma 0.0 \
        -omega 30 -omega_sigma 0.0
done

sem --wait
echo "MCMC is finished!"
