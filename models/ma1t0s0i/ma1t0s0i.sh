#!/bin/bash

# Ensure that we have only one or two arguments passed.
if [[ "$#" -ne 1 ]] && [[ "$#" -ne 2 ]]; then
    echo "Usage: ma1t0s0i.sh [results database] [observed database]"
    exit 1
fi
SCRIPT_DIR=$(dirname "$0")

i=1  # Setup our progress bar.
DURATION=100
already_done() { for ((done=0; done < $i; done++)); do printf "▉"; done }
remaining() { for ((remain=$i; remain < ${DURATION}; remain++)); do printf " "; done }
percentage() { printf "| %s%%" $(( ($i*100)/${DURATION}*100/100 )); }
clean_line() { printf "\r"; }

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
already_done; remaining; percentage
python3 ${SCRIPT_DIR}/ma1t0s0i.py \
    -mdb "$1" \
    -simulation_n 100 \
    -epsilon 0.55 \
    -delta_f cosine \
    -iterations_n 1000 \
    -flush_n 500 \
    -uid ${sample_uids} \
    -loci ${sample_loci} \
    -n 50 -n_sigma 0.0 \
    -f 100 -f_sigma 0.0 \
    -c 0.01067 -c_sigma 0.01003 \
    -d 0.00255 -d_sigma 0.00108 \
    -kappa 3 -kappa_sigma 0.0 \
    -omega 30 -omega_sigma 0.0
clean_line

# Repeat 99 more times.
for ((i=2; i<${DURATION}; i++)); do
    already_done; remaining; percentage
    python3 ${SCRIPT_DIR}/ma1t0s0i.py \
        -mdb "$1" \
        -simulation_n 100 \
        -epsilon 0.55 \
        -delta_f cosine \
        -iterations_n 1000 \
        -flush_n 500 \
        -uid ${sample_uids} \
        -loci ${sample_loci} \
        -n_sigma 0.0 \
        -f_sigma 0.0 \
        -c_sigma 0.01003 \
        -d_sigma 0.00108 \
        -kappa_sigma 0.0 \
        -omega_sigma 0.0
    clean_line
done

printf "\nMCMC is finished!"
