#!/bin/bash

# Perform one run of MCMC for each sample found in our real database in series.
for r in $(sqlite3 data/real.db "SELECT DISTINCT SAMPLE_UID, LOCUS FROM REAL_ELL WHERE LOCUS LIKE 'D20S481'"); do
    IFS='|' read -r -a array <<< "$r"

    # Run the MCMC.
    python3 emcee.py \
        -edb "data/${array[0]}-${array[1]}-emcee.db" \
    	-r 50 \
        -epsilon 0.8 \
    	-it 100000 \
    	-rsu "${array[0]}" \
    	-l "${array[1]}" \
    	-i_0 18 -i_0_sigma 1.0 \
    	-big_n 1000 -big_n_sigma 0.0 \
    	-mu 0.001 -mu_sigma 0.001 \
    	-s 0.05 -s_sigma 0.001 \
    	-kappa 3 -kappa_sigma 0.0 \
    	-omega 100 -omega_sigma 0.0 \
    	-u 0.90 -u_sigma 0.001 \
    	-v 0.03 -v_sigma 0.001 \
    	-m 0.9 -m_sigma 0.001 \
    	-p 0.001 -p_sigma 0.0005

#    # Delete the burn-in period. Deleting first 5000 entries.
#    sqlite3 data/${array[0]}-${array[1]}-emcee.db "DELETE FROM WAIT_POP WHERE ROWID IN (
#            SELECT ROWID FROM WAIT_POP
#            WHERE REAL_SAMPLE_UID LIKE '${array[0]}' AND REAL_LOCUS LIKE '${array[1]}'
#            ORDER BY ROWID ASC
#            LIMIT 5000
#        );"
done
echo "MCMC runs for each real sample is finished."
sleep 1

# Create the head database if it does not exist. We rip the schema off an existing database.
if [ ! -f data/emcee.db ]; then
    cp $(find data/ -name "*emcee.db" | head -1) data/emcee.db
    sqlite3 data/emcee.db "DELETE FROM WAIT_POP WHERE 1 = 1;"
    sleep 1
fi

# Iterate throguh each of the generated databases, and copy this table to the head. This must be sequential.
for f in data/*-emcee.db; do
    sqlite3 data/emcee.db "ATTACH '${f}' AS A; INSERT INTO WAIT_POP SELECT * FROM A.WAIT_POP; DETACH DATABASE A;"
done
echo "Database merging is finished."
sleep 1

# Delete the emcee database files (dangerous operation, left commented out for now).
for f in data/*-emcee.db; do
   rm ${f}
done
echo "Removed generated database files."

