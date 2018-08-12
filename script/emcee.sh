#!/bin/bash

# Run our MCMC 3 times, we compare the distributions of each 3 to verify convergence manually.
for run in 1 2 3; do

    # Our criteria for the loci and sample IDs to use for this run of MCMC.
    sample_uids=(); loci=()
    for r in $(sqlite3 data/real.db "SELECT DISTINCT SAMPLE_UID, LOCUS FROM REAL_ELL WHERE LOCUS LIKE 'D20S481'"); do
        IFS='|' read -r -a array <<< "$r"
        sample_uids+="${array[0]} "; loci+="${array[1]} "
    done

    # Run the MCMC.
    python3 emcee.py \
        -edb "data/emcee-${run}.db" \
        -r 200 \
        -epsilon 0.8 \
        -it 10000 \
        -rsu ${sample_uids} \
        -l ${loci} \
        -i_0 18 -i_0_sigma 1.0 \
        -big_n 1000 -big_n_sigma 0.0 \
        -mu 0.001 -mu_sigma 0.01 \
        -s 0.05 -s_sigma 0.01 \
        -kappa 3 -kappa_sigma 0.0 \
        -omega 100 -omega_sigma 0.0 \
        -u 0.90 -u_sigma 0.01 \
        -v 0.03 -v_sigma 0.01 \
        -m 0.9 -m_sigma 0.01 \
        -p 0.001 -p_sigma 0.005

    ## Delete the burn-in period. Deleting first 5000 entries.
    #sqlite3 data/${array[0]}-${array[1]}-emcee.db "DELETE FROM WAIT_POP WHERE ROWID IN (
    #        SELECT ROWID FROM WAIT_POP
    #        WHERE REAL_SAMPLE_UID LIKE '${array[0]}' AND REAL_LOCUS LIKE '${array[1]}'
    #        ORDER BY ROWID ASC
    #        LIMIT 5000
    #    );"

done
echo "MCMC is finished!"
