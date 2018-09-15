#!/bin/bash

# Run our MCMC 3 times, we compare the distributions of each 3 to verify convergence manually.
for run in 1 2 3; do

    # Our criteria for the loci and sample IDs to use for this run of MCMC.
    sample_uids=(); loci=()
    for r in $(sqlite3 data/real.db "SELECT DISTINCT SAMPLE_UID, LOCUS FROM REAL_ELL"); do
        IFS='|' read -r -a array <<< "$r"
        sample_uids+="${array[0]} "; loci+="${array[1]} "
    done

    # Run the MCMC. We are now only focusing on PL1 model from Sainudiin paper.
    python3 model.py \
        -edb "data/posterior-${run}-abc.db" \
        -type "ABC" \
        -rs 200 \
        -rp 10 \
        -epsilon 0.50 \
        -it 100000 \
        -rsu ${sample_uids} \
        -l ${loci} \
        -big_n 1000 -big_n_sigma 0.0 \
        -mu 0.001 -mu_sigma 0.001 \
        -s 0.05 -s_sigma 0.001 \
        -kappa 3 -kappa_sigma 0.0 \
        -omega 100 -omega_sigma 0.0 \
        -u 0.90 -u_sigma 0.001 \
        -v 0.03 -v_sigma 0.001 \
        -m 1 -m_sigma 0.0 \
        -p 0.0 -p_sigma 0.0

done
echo "MCMC is finished!"
