#!/bin/bash

# Perform the sampling for each population found in our simulated database in parallel.
compare(){
    python3 src/compare.py \
        -ssdb "data/$1-compare.db" \
        -r 30 \
        -sei "$1" \
        -rsu "SA000288S" \
        -l "D20S481"
}
export -f compare; parallel "compare {}" ::: $(sqlite3 data/simulated.db "SELECT EFF_ID FROM EFF_POP;")
echo "Sampling is finished."

# Create the head database. We rip the schema off of an existing database.
cp $(find data/ -name "*compare.db" | head -1) data/compare.db
sqlite3 data/compare.db "DELETE FROM DELTA_POP WHERE 1 = 1;"

# Iterate through each of the generated databases, and copy this table to the head. This must be sequential.
for f in data/*-compare.db; do
	sqlite3 data/compare.db "ATTACH '${f}' AS A; INSERT INTO DELTA_POP SELECT * FROM A.DELTA_POP; DETACH DATABASE A;"
done
echo "Database merging is finished."

# Delete the sample database files (dangerous operation, left commented out for now).
#for f in data/*-compare.db; do
#	rm ${f}
#done
#echo "Removed generated database files."

