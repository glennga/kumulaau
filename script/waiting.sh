#!/bin/bash

# Visualize all instances of our MCMC waiting times.
for f in data/model-*.db; do
    python3 src/waiting.py \
        -db "$f" \
        -f 1 \
        -image "${f%".db"}.png" \
        0 0 0.0005 0.001 0.0005 0 0
done
