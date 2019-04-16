#!/bin/bash

set -e
SCRIPT_DIR=$(dirname "$0")

i=1  # Setup our progress bar.
already_done() { for ((done=0; done < $i; done++)); do printf "▉"; done }
remaining() { for ((remain=$i; remain < ${MCMC_LINKS}; remain++)); do printf " "; done }
percentage() { printf "| #${j:-1}, %s%%" $(( ($i*100)/${MCMC_LINKS}*100/100 )); }
clean_line() { printf "\r"; }

# Run once to seed our database. Must break into parts because GC is garbage ):<
already_done; remaining; percentage
python3 ${SCRIPT_DIR}/abc1t0s0i.py \
    -mdb "$1" \
    -observations "${OBSERVATIONS}" \
    -simulation_n ${SIMULATION_N} \
    -epsilon ${EPSILON} \
    -delta_f ${DELTA_F} \
    -iterations_n ${ITERATIONS_N} \
    -flush_n ${FLUSH_N} \
    -n_start ${N_START} -n_sigma ${N_SIGMA} \
    -f_start ${F_START} -f_sigma ${F_SIGMA} \
    -c_start ${C_START} -c_sigma ${C_SIGMA} \
    -d_start ${D_START} -d_sigma ${D_SIGMA} \
    -kappa_start ${KAPPA_START} -kappa_sigma ${KAPPA_SIGMA} \
    -omega_start ${OMEGA_START} -omega_sigma ${OMEGA_SIGMA}
clean_line

# Repeat MCMC_LINKS - 1 more times.
for ((i=2; i<${MCMC_LINKS}; i++)); do
    already_done; remaining; percentage
    python3 ${SCRIPT_DIR}/abc1t0s0i.py \
        -mdb "$1" \
        -observations "${OBSERVATIONS}" \
		-simulation_n ${SIMULATION_N} \
		-epsilon ${EPSILON} \
		-delta_f ${DELTA_F} \
		-iterations_n ${ITERATIONS_N} \
		-flush_n ${FLUSH_N} \
		-n_sigma ${N_SIGMA} \
		-f_sigma ${F_SIGMA} \
		-c_sigma ${C_SIGMA} \
		-d_sigma ${D_SIGMA} \
		-kappa_sigma ${KAPPA_SIGMA} \
		-omega_sigma ${OMEGA_SIGMA}
    clean_line
done
printf "\n"
