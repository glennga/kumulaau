#!/bin/bash

# Visualize all instances of our MCMC waiting times and posterior.
for f in data/method*.db; do
    python3 src/plot.py \
        -db "$f" \
        -function 1 \
        -image "${f%".db"}-waiting.png" \
        -param 0 0 0.0005 0.001 0.0001 0 0

    python3 src/plot.py \
        -db "$f" \
        -function 2 \
        -image "${f%".db"}-posterior.png" \
        -param 0 0 0.0005 0.001 0.0001 0 0

    python3 src/plot.py \
        -db "$f" \
        -function 3 \
        -image "${f%".db"}-trace.png" 
done
