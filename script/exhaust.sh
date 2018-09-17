#!/bin/bash

# Run our exhaustive search over each distinct parameter 10-tuple.
python3 src/exhaust.py \
   -i_0 15 16 17 18 \
   -big_n 1000 \
   -mu 0.01 0.001 0.02 \
   -s 0.01 0.02 0.001 \
   -kappa 1 \
   -omega 30 \
   -u 0.6  0.5 0.55 \
   -v 0.005 0.01 0.008 \
   -m 0.5 0.02 0.03 \
   -p 0.5 0.4 0.6 \
   -r 10

