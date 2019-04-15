#!/bin/bash
set -e

# Ensure that we have only one or two arguments passed.
if [[ "$#" -ne 1 ]] && [[ "$#" -ne 2 ]]; then
    echo "Usage: ele4t1s2i.sh [results database] [observed database]"
    exit 1
fi
SCRIPT_DIR=$(dirname "$0")

i=1  # Setup our progress bar.
DURATION=100
already_done() { for ((done=0; done < $i; done++)); do printf "▉"; done }
remaining() { for ((remain=$i; remain < ${DURATION}; remain++)); do printf " "; done }
percentage() { printf "| %s%%" $(( ($i*100)/${DURATION}*100/100 )); }
clean_line() { printf "\r"; }

# Our criteria for the loci and sample IDs to use for this run of MCMC: The Italian populace.
sample_uids=(); sample_loci=()
for r in $(sqlite3 ${2:-data/observed.db} "SELECT DISTINCT SAMPLE_UID, LOCUS \
                                           FROM OBSERVED_ELL \
                                           WHERE POP_NAME LIKE 'Italians';"); do
    IFS='|' read -r -a array <<< "$r"
    sample_uids+="${array[0]} "; sample_loci+="${array[1]} "
done

# Run once to seed our database. Must break into parts because GC is garbage ):<
already_done; remaining; percentage
python3 ${SCRIPT_DIR}/ele4t1s2i.py \
	-mdb "$1" \
    -simulation_n 100 \
    -r 0.5 \
    -bin_n 500 \
    -iterations_n 1000 \
    -flush_n 500 \
    -uid ${sample_uids} \
    -loci ${sample_loci} \
    -delta_f cosine \
    -n_b 10 -n_b_sigma 5.0 \
    -n_s1 10 -n_s1_sigma 5.0 \
    -n_s2 10 -n_s2_sigma 5.0 \
    -n_e 100 -n_e_sigma 5.0 \
    -f_b 100 -f_b_sigma 5.0 \
    -f_s1 100 -f_s1_sigma 5.0 \
    -f_e 100 -f_e_sigma 5.0 \
	-alpha 0.1 -alpha_sigma 0.01 \
    -c 0.00600 -c_sigma 0.0 \
    -d 0.00046 -d_sigma 0.0 \
    -kappa 3 -kappa_sigma 0.0 \
    -omega 30 -omega_sigma 0.0
clean_line

# Repeat 9 more times.
for i in {2..10}; do
	already_done; remaining; percentage
    python3 ${SCRIPT_DIR}/ele4t1s2i.py \
		-mdb "$1" \
		-simulation_n 100 \
    	-r 0.5 \
    	-bin_n 500 \
		-iterations_n 1000 \
		-flush_n 500 \
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
    clean_line
done

printf "\nMCMC is finished!\n"
