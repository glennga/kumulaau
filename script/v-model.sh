#!/bin/bash

for f in data/posterior-*.db; do
    python3 visual.py \
        -db "$f" \
        -f "posterior" \
        -image "${f%".db"}.png" \
        -hs 0.001 0.001 0.001 0.001 0.001 0.0005
done

