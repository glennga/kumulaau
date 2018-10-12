#!/bin/bash

# Run our MCMC 3 times, we compare the distributions of each 3 to verify convergence manually.
for run in 1 2 3; do

    # Our criteria for the loci and sample IDs to use for this run of MCMC.
    sample_uids=(); loci=()
    for r in $(sqlite3 data/observed.db "SELECT DISTINCT SAMPLE_UID, LOCUS FROM OBSERVED_ELL"); do
        IFS='|' read -r -a array <<< "$r"
        sample_uids+="${array[0]} "; loci+="${array[1]} "
    done

    # Run the MCMC.
    python3 src/model.py \
        -mdb "data/model-${run}.db" \
        -simulation_n 10 \
        -sample_n 10 \
        -epsilon 0.60 \
        -iterations_n 10000 \
        -uid_observed ${sample_uids} \
        -locus_observed ${loci} \
        -n 1000 -n_sigma 0.0 \
        -f 1 -f_sigma 0.0 \
        -c 0.001 -c_sigma 0.001 \
        -u 1.2 -u_sigma 0.001 \
        -d 0.001 -d_sigma 0.001 \
        -kappa 3 -kappa_sigma 0.0 \
        -omega 100 -omega_sigma 0.0

done
echo "MCMC is finished!"
