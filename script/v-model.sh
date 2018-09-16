#!/bin/bash

# Visualize all instances of our mutation model parameter estimation.
for f in data/model-*.db; do
    python3 visual.py \
        -db "$f" \
        -f "model" \
        -image "${f%".db"}.png" \
        -hs 0.001 0.001 0.001 0.001 0.001 0.0005
done

