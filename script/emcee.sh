#!/bin/bash

# Perform one run of MCMC (metropolis-hastings).
python3 emcee.py \
	-r 30 \
	-it 100000 \
	-rsu "SA000288S" \
	-l "D20S481" \
	-i_0 25 -i_0_sigma 1.5 \
	-big_n 1000 -big_n_sigma 0.0 \
	-mu 0.025 -mu_sigma 0.0001 \
	-s 0.01 -s_sigma 0.001 \
	-kappa 3 -kappa_sigma 0.0 \
	-omega 50 -omega_sigma 0.0 \
	-u 0.90 -u_sigma 0.001 \
	-v 0.04 -v_sigma 0.001 \
	-m 0.52 -m_sigma 0.001 \
	-p 0.25 -p_sigma 0.001

