#!/bin/bash

python_script="experiments.py"

#folder="test_cases_correct/"
folder="test_cases_incorrect/"
#files=( "deeppoly_relu" "deeppoly_affine" "deeppoly_maxpool" "zono_relu" "zono_affine" "zono_maxpool" "refinezono_relu" "refinezono_affine" "refinezono_maxpool" "ibp_relu" "ibp_affine" "ibp_maxpool" "polyzono_relu" "polyzono_affine" "polyzono_maxpool" "fb_relu" "fb_affine" "fb_maxpool" )

combinations=("1 1 1" "2 2 1" "2 3 2" "3 3 2" "3 4 2" "4 4 3" "5 5 4" "5 6 5" "7 7 7" "7 8 7" "8 8 8" "9 10 8" "10 10 5" "15 15 5" "20 20 5" )
#combinations=("1 1 1" "2 2 1" "2 3 2" "3 3 2" "3 4 2" "4 4 3" "5 5 4" "5 6 5" "7 7 7" "7 8 7" "8 8 8" "9 10 8")


timeout_value=300

for f in "${files[@]}"
do
    for c in "${combinations[@]}"
    do
        #timeout $timeout_value python "$python_script" "${folder}${f}" "${c}"
        

        output=$(time -p (timeout $timeout_value python "$python_script" "${folder}${f}" "${c[@]}") 2>&1)
        exit_status=$?
        execution_time=$(echo "$output" | awk '/^real/ {print $2}')
        

        # Print the execution time
        echo "${folder}${f} ${c[@]}: $execution_time seconds"

        if [ $exit_status -eq 124 ]; then
            echo "Timeout occurred for parameters: ${folder} ${f} ${c}"
        fi
    done
done