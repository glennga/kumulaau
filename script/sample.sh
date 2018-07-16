#!/bin/bash

# Perform the sampling for each population found in our simulated database.
sqlite3 data/simulated.db "SELECT EFF_ID FROM EFF_POP;" \
    | while read -r line ; do
    python3 sample.py \
        -ssdb "data/$line-sample.db" \
        -r 30 \
        -sei "$line" \
        -rsu "SA000288S" \
        -l "D20S481"
done
echo "Sampling is finished."

# Create the head database. We rip the schema off of an existing database.
cp $(find data/ -name "*sample.db" | head -1) data/sample.db
sqlite3 data/sample.db "DELETE FROM DELTA_POP WHERE 1 = 1;"

# Iterate through each of the generated databases, and copy this table to the head.
for f in data/*-sample.db; do
	sqlite3 data/sample.db "ATTACH '${f}' AS A; INSERT INTO DELTA_POP SELECT * FROM A.DELTA_POP; DETACH DATABASE A;"
done
echo "Database merging is finished."

# Delete the sample database files (dangerous operation, left commented out for now).
#for f in data/*-sample.db; do
#	rm ${f}
#done
#echo "Removing generated database files."
