#!/bin/bash

# Perform one run of MCMC (metropolis-hastings).
python3 emcee.py \
	-r 10 \
	-it 10000 \
	-rsu "SA000288S" \
	-l "D20S481" \
	-i_0 15 -i_0_sigma 2.0 \
	-big_n 1000 -big_n_sigma 0.0 \
	-mu 0.001 -mu_sigma 0.001 \
	-s 0.001 -s_sigma 0.001 \
	-kappa 3 -kappa_sigma 0.0 \
	-omega 30 -omega_sigma 0.0 \
	-u 0.5 -u_sigma 0.1 \
	-v 0.001 -v_sigma 0.001 \
	-m 0.5 -m_sigma 0.01 \
	-p 0.5 -p_sigma 0.01
