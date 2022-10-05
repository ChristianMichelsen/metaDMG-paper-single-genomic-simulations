
# seeds=`seq 1 2`
declare -a damages=("0.0" "0.014" "0.047" "0.138" "0.303" "0.466" "0.96")
declare -a species=("homo" "betula")


## now loop through the above array
for specie in "${species[@]}"
do
    for damage in "${damages[@]}"
    do
        mkdir -p ./data/results-$specie-$damage
        cp ./data/results/sim-$specie-$damage-*.results.parquet ./data/results-$specie-$damage

    done
done