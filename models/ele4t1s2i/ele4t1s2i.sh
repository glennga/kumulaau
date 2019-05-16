#!/bin/bash

set -e
SCRIPT_DIR=$(dirname "$0")

i=1  # Setup our progress bar.
already_done() { for ((done=0; done < $(( ($i*100 / ${MCMC_LINKS}*100)*70/10000 )); done++)); do printf "▓"; done }
remaining() { for ((remain=$(( ($i*100 / ${MCMC_LINKS}*100)*70/10000 )); remain < 70; remain++)); do printf " "; done }
percentage() { printf "| #${j:-1}, %s%%" $(( ($i*100)/${MCMC_LINKS}*100/100 )); }
clean_line() { printf "\r"; }

# Run once to seed our database. Must break into parts because GC is garbage ):<
already_done; remaining; percentage
python3 ${SCRIPT_DIR}/ele4t1s2i.py \
	-mdb "${MDB}" \
	-observations "${OBSERVATIONS}" \
	-simulation_n ${SIMULATION_N} \
	-r ${R} \
	-bin_n ${BIN_N} \
	-delta ${DELTA} \
	-iterations_n ${ITERATIONS_N} \
	-flush_n ${FLUSH_N} \
	-i_0_start ${I_0_START} -i_0_sigma ${I_0_SIGMA} \
	-n_b_start ${N_B_START} -n_b_sigma ${N_B_SIGMA} \
	-n_s1_start ${N_S1_START} -n_s1_sigma ${N_S1_SIGMA} \
	-n_s2_start ${N_S2_START} -n_s2_sigma ${N_S2_SIGMA} \
	-n_e_start ${N_E_START} -n_e_sigma ${N_E_SIGMA} \
	-f_b_start ${F_B_START} -f_b_sigma ${F_B_SIGMA} \
	-f_s1_start ${F_S1_START} -f_s1_sigma ${F_S1_SIGMA} \
	-f_s2_start ${F_S2_START} -f_s2_sigma 0.0 \
	-f_e_start ${F_E_START} -f_e_sigma ${F_E_SIGMA} \
	-alpha_start ${ALPHA_START} -alpha_sigma ${ALPHA_SIGMA} \
	-c_start ${C_START} -c_sigma ${C_SIGMA} \
	-d_start ${D_START} -d_sigma ${D_SIGMA} \
	-kappa_start ${KAPPA_START} -kappa_sigma ${KAPPA_SIGMA} \
	-omega_start ${OMEGA_START} -omega_sigma ${OMEGA_SIGMA}
clean_line

# Repeat MCMC_LINKS - 1 more times.
for ((i=2; i<${MCMC_LINKS}; i++)); do
    already_done; remaining; percentage
    python3 ${SCRIPT_DIR}/ele4t1s2i.py \
		-mdb "${MDB}" \
		-observations "${OBSERVATIONS}" \
		-simulation_n ${SIMULATION_N} \
		-r ${R} \
		-bin_n ${BIN_N} \
		-delta ${DELTA} \
		-iterations_n ${ITERATIONS_N} \
		-i_0_sigma ${I_0_SIGMA} \
		-n_b_sigma ${N_B_SIGMA} \
		-n_s1_sigma ${N_S1_SIGMA} \
		-n_s2_sigma ${N_S2_SIGMA} \
		-n_e_sigma ${N_E_SIGMA} \
		-f_b_sigma ${F_B_SIGMA} \
		-f_s1_sigma ${F_S1_SIGMA} \
		-f_s2_sigma 0.0 \
		-f_e_sigma ${F_E_SIGMA} \
		-alpha_sigma ${ALPHA_SIGMA} \
		-c_sigma ${C_SIGMA} \
		-d_sigma ${D_SIGMA} \
		-kappa_sigma ${KAPPA_SIGMA} \
		-omega_sigma ${OMEGA_SIGMA}
    clean_line
done

# Finish the output.
already_done; remaining; percentage; sleep 1; clean_line;
