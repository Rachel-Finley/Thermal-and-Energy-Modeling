#!/bin/bash

folder_name="$1"
benchmark_name="$2"

mkdir -p diagnostic_data/"$folder_name"

./cpu_util.sh > diagnostic_data/"$folder_name"/cpu_util.txt &
./cpu_temp.sh > diagnostic_data/"$folder_name"/cpu_temp.txt &
nvidia-smi -l 1 > diagnostic_data/"$folder_name"/gpu_status.txt &
./disk_temp_monitor.sh > diagnostic_data/"$folder_name"/disk_temp.txt &
./disk_util_monitor.sh > diagnostic_data/"$folder_name"/disk_util.txt &

if [ "$3" == "no" ]; then
    echo "Experiment: $folder_name" >> diagnostic_data/"$folder_name"/start_end_time.txt
    echo "Starts at" >> diagnostic_data/"$folder_name"/start_end_time.txt
    date >> diagnostic_data/"$folder_name"/start_end_time.txt
    python3 "$benchmark_name"
    echo "Ends at" >> diagnostic_data/"$folder_name"/start_end_time.txt
    date >> diagnostic_data/"$folder_name"/start_end_time.txt

elif [ "$3" == "yes" ]; then
    for (( i=0; i<$4; i++ )); do
        echo "Iteration $i starts at: " >> diagnostic_data/"$folder_name"/start_end_time.txt
        date >> diagnostic_data/"$folder_name"/start_end_time.txt
        python3 "$benchmark_name"
        #./"$benchmark_name"
        echo "Iteration $i ends at: " >> diagnostic_data/"$folder_name"/start_end_time.txt
        date >> diagnostic_data/"$folder_name"/start_end_time.txt
        sleep "$5"m
    done
fi


# python3 PROCESS_DIAGNOSTIC_TXT_FILES.py
# python3 CREATE_CPU_SUMMARY_STATISTICS.py


