#!/bin/bash

python_script="experiments_int.py"

folder="test_cases_correct/"
times="pz_affine/"
#folder="test_cases_incorrect/"
#files=( "deeppoly_relu" "deeppoly_affine" "deeppoly_maxpool" "zono_relu" "zono_affine" "zono_maxpool" "refinezono_relu" "refinezono_affine" "refinezono_maxpool" "ibp_relu" "ibp_affine" "ibp_maxpool" "polyzono_relu" "polyzono_affine" "polyzono_maxpool" "fb_relu" "fb_affine" "fb_maxpool" )
#files=( "deeppoly_relu" "deeppoly_affine" "deeppoly_maxpool" "zono_relu" "zono_affine" "refinezono_relu" "refinezono_affine" "ibp_relu" "ibp_affine" "ibp_maxpool" "polyzono_relu" "polyzono_affine" "fb_relu" "fb_affine" "fb_maxpool" )
#combinations=("1 1 2" "1 4 2" "1 7 2" "1 10 2" "3 4 2" "3 7 2" "3 10 2" "4 7 2" "4 10 2" "5 7 2" "5 10 2" "7 7 2" "7 10 2" "8 10 2" "10 10 2")
#combinations=("1 3 1" "3 3 3" "5 3 5" "7 3 7" "9 3 9")
#files2=("deeppoly_maxpool" "zono_affine" "zono_maxpool" "ibp_affine" "ibp_maxpool" "polyzono_maxpool")
#files=( "deeppoly_affine" "polyzono_affine" )
files=("polyzono_affine")
#nsym=5
timeout_value=320

for f in "${files[@]}"
do
    #for c in "${combinations[@]}"
    for i in {1..8}
    do
        for j in {1..8}
        do
            output=$(time -p (timeout $timeout_value python "$python_script" "${folder}${f}" ${i} ${j}) > "${times}out${f} ${i} ${j}" 2>&1)
            exit_status=$?
            #execution_time=$(echo "$output" | awk '/^real/ {print $2}')
            

            # Print the execution time
            #echo "${folder}${f} ${c[@]}: $execution_time seconds"
            if [ $exit_status -eq 124 ]; then
                echo "Timeout occurred for parameters: ${folder} ${f} ${i} ${j}"
                break
            fi
        done
    done
done

# for f in "${files2[@]}"
# do
#     #for c in "${combinations[@]}"
#     for i in {1..30}
#     do
#         output=$(time -p (timeout $timeout_value python "$python_script" "${folder}${f}" ${i} ${nsym}) > "${times}out${f} ${i} ${nsym}" 2>&1)
#         exit_status=$?
#         #execution_time=$(echo "$output" | awk '/^real/ {print $2}')
        

#         # Print the execution time
#         #echo "${folder}${f} ${c[@]}: $execution_time seconds"
#         if [ $exit_status -eq 124 ]; then
#             echo "Timeout occurred for parameters: ${folder} ${f} ${i} ${nsym}"
#             break
#         fi
#     done
# done