#!/bin/bash

# Check if mode is passed as an argument (1: no profiling, 2: nvprof, 3: nvprof --profile-from-start off)
if [ $# -lt 1 ]; then
    echo "Usage: $0 <mode>"
    echo "Mode options: "
    echo "1: No profiling"
    echo "2: nvprof"
    echo "3: nvprof --profile-from-start off"
    exit 1
fi

mode=$1

# Step 1: Build the CUDA program if necessary using Makefile
echo "Checking if recompilation is necessary..."
make

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Step 2: Poll nvidia-smi until GPU utilization is at 0%
while true; do
    # Get GPU utilization percentage using nvidia-smi
    utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{print $1}')

    # Check if utilization is 0
    if [ "$utilization" -eq 0 ]; then
        echo "GPU is idle, running the program..."

        # Step 3: Execute the program based on mode
        case "$mode" in
            1)
                echo "Running ./test ./params/30epoch without profiling"
                ./test ./params/30epoch
                ;;
            2)
                echo "Running with nvprof"
                nvprof ./test ./params/30epoch
                ;;
            3)
                echo "Running with nvprof --profile-from-start off"
                nvprof --profile-from-start off ./test ./params/30epoch
                ;;
            *)
                echo "Invalid mode. Exiting."
                exit 1
                ;;
        esac

        break
    else
        echo "GPU is busy (utilization: $utilization%), waiting..."
        sleep 5
    fi
done
