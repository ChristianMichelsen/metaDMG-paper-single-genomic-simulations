
# seeds=`seq 1 2`
declare -a damages=("0" "014" "047" "14" "30" "46" "615" "93")


## now loop through the above array
for damage in "${damages[@]}"
    mkdir ./data/results-$damage
    cp ./data/results/sim-$damage-*.parquet ./data/results-$damage
do
