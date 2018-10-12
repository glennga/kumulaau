#!/bin/bash

# Visualize all instances of our MCMC waiting times and posterior.
for f in data/model-*.db; do
    python3 src/plot.py \
        -db "$f" \
        -function 1 \
        -image "${f%".db"}-waiting.png" \
        0 0 0.0005 0.001 0.0005 0 0

    python3 src/plot.py \
        -db "$f" \
        -function 1 \
        -image "${f%".db"}-posterior.png" \
        0 0 0.0005 0.001 0.0005 0 0
done
