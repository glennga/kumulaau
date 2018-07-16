#!/bin/bash

sqlite3 data/simulated.db "SELECT EFF_ID FROM EFF_POP " \
    | while read -r sei ; do
    echo $(sei)
#    python3 sample.py \
#        -ssdb "data/$(sei)-sample.db"
#        -r 100 \
#        -sei "$sei" \
#        -rsu "SA000288S" \
#        -l "D20S481"
done
