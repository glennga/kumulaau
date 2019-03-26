#!/bin/bash

# Ensure that we have only one or two arguments passed.
if [[ "$#" -ne 1 ]] && [[ "$#" -ne 2 ]]; then
    echo "Usage: ma4t1s2i.sh [results database] [observed database]"
    exit 1
fi
SCRIPT_DIR=$(dirname "$0")

# Our criteria for the loci and sample IDs to use for this run of MCMC: The Colombian populace.
sample_uids=(); sample_loci=()
for r in $(sqlite3 ${2:-data/observed.db} "SELECT DISTINCT SAMPLE_UID, LOCUS \
                                           FROM OBSERVED_ELL \
                                           WHERE LOCUS LIKE 'D16S539' \
                                           AND POP_UID LIKE 'PO000503I';"); do
    IFS='|' read -r -a array <<< "$r"
    sample_uids+="${array[0]} "; sample_loci+="${array[1]} "
done

# Run once to seed our database. Must break into parts because GC is garbage ):<
python3 ${SCRIPT_DIR}/ma4t1s2i.py \
	-mdb "$1" \
    -simulation_n 100 \
    -epsilon 0.55 \
    -iterations_n 10000 \
    -flush_n 5000 \
    -uid ${sample_uids} \
    -loci ${sample_loci} \
    -delta_f cosine \
    -n_b 10 -n_b_sigma 5.0 \
    -n_s1 10 -n_s1_sigma 5.0 \
    -n_s2 10 -n_s2_sigma 5.0 \
    -f_b 100 -f_b_sigma 5.0 \
    -f_s1 100 -f_s1_sigma 5.0 \
    -f_e 100 -f_e_sigma 5.0 \
	-alpha 0.1 -alpha_sigma 0.01 \
    -c 0.01067 -c_sigma 0.0 \
    -d 0.00255 -d_sigma 0.0 \
    -kappa 3 -kappa_sigma 0.0 \
    -omega 30 -omega_sigma 0.0
echo "MCMC Progress [1/10]."

# Repeat 9 more times.
for i in {2..10}; do
    python3 ${SCRIPT_DIR}/ma4t1s2i.py \
		-mdb "$1" \
		-simulation_n 100 \
		-epsilon 0.55 \
		-iterations_n 10000 \
		-flush_n 5000 \
		-uid ${sample_uids} \
		-loci ${sample_loci} \
		-delta_f cosine \
		-n_b_sigma 5.0 \
		-n_s1_sigma 5.0 \
		-n_s2_sigma 5.0 \
		-f_b_sigma 5.0 \
		-f_s1_sigma 5.0 \
		-f_e_sigma 5.0 \
		-alpha_sigma 0.01 \
		-c_sigma 0.0 \
		-d_sigma 0.0 \
		-kappa_sigma 0.0 \
		-omega_sigma 0.0
    echo "MCMC Progress [$i/10]."
done

echo "MCMC is finished!"
