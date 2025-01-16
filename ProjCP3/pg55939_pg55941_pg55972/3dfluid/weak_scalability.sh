#!/bin/bash

# Range of sizes
START=32
END=320
STEP=32

# Loop through the range
for ((SIZE=$START; SIZE<=$END; SIZE+=$STEP)); do
    echo "Testing with SIZE=$SIZE"
    
    # Compile with the current SIZE
    make release SIZE=$SIZE
    
    make run

    echo "=================================================================================="
done

echo "All tests completed."
