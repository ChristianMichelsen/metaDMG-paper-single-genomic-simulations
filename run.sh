# bowtie2-build Mycobacterium_leprae.fa Mycobacterium_leprae.fa

species=$1
# species="homo"
# species="betula"

quick=true
quick=false
threads=10


if [[ "$species" == "homo" ]]; then
    genome=./genome/NC_012920.1_genomic.fna
elif [[ "$species" == "betula" ]]; then
    genome=./genome/KX703002.1.fa
else
    echo "bad species."
    exit 1
fi


if [ "$quick" = true ] ; then
    declare -a damages=("0.0" "0.96")
    declare -a Nreads=("100" "1000")
    declare -a length_means=("60")
    seeds=`seq 0 1`
else
    declare -a damages=("0.0" "0.014" "0.047" "0.138" "0.303" "0.466" "0.96")
    declare -a Nreads=("25" "50" "75" "100" "200" "300" "400" "500" "750" "1000" "5000" "10000" "100000")
    declare -a length_means=("35" "60" "90")
    seeds=`seq 0 99`
fi

length_std=10
quality_scores=Test_Examples/AccFreqL150R1.txt

if [ ! -f $genome.1.bt2 ]; then
        bowtie2-build $genome $genome
fi


mkdir -p fastq
mkdir -p bam

COUNTER=0


function compute_lognormal_mean {
    mu=$length_mean
    sigma=$length_std
    echo "l($mu^2 / sqrt($sigma^2 + $mu^2))" | bc -l | awk '{printf "%f", $0}'
}

function compute_lognormal_std {
    mu=$length_mean
    sigma=$length_std
    echo "sqrt(l(1 + $sigma^2 / $mu^2))" | bc -l | awk '{printf "%f", $0}'
}


function simulate_fastq {


    if ! [[ -f $fastq.fq  &&  -s $fastq.fq  ]] # does not exist or is empty
    then
        if (( $(echo "$damage > 0.0001" |bc -l) ));
        then
            briggs="-b 0.024,0.36,$damage,0.0097"
        else
            briggs="--noerror"
        fi

        lognorm_mean=$(compute_lognormal_mean)
        lognorm_std=$(compute_lognormal_std)

        args="-i $genome -t $threads -r $Nread -ld LogNorm,$lognorm_mean,$lognorm_std -s $seed -seq SE -f fq -q1 $quality_scores $briggs -o $fastq"
        ./ngsngs $args
        # echo $args
    fi
}

function make_bam_file {

    if ! [[ -f $bam  &&  -s $bam  ]] # does not exist or is empty
    then
        simulate_fastq
        bowtie2 -x $genome -q $fastq.fq --threads $threads --no-unal | samtools view -bS -@ $threads - | samtools sort -n -@ $threads -> $bam
    fi

}


for damage in "${damages[@]}"
do

    for Nread in "${Nreads[@]}"
    do

        for length_mean in "${length_means[@]}"
        do

            for seed in $seeds;
            do

                basename=sim-$species-$damage-$Nread-$length_mean-$seed
                fastq=./fastq/$basename
                bam=./bam/$basename.bam

                # echo "$damage, $Nread, $seed"
                make_bam_file
                let COUNTER++

            done
        done
    done
done

echo $COUNTER

#%%

# metaDMG config ./bam/*.bam --damage-mode local --bayesian --long-name --parallel-samples $threads --overwrite
# metaDMG compute config.yaml


# mkdir -p mapDamage
# mapDamage -i $bamfile -r $db --no-stats -d mapDamage/$(basename $file).$(basename $db).AFG --merge-libraries
# ./metaDMG-cpp getdamage --minlength 10 --printlength 50 --threads 8 $bam |  cut -f7-8 | sed -n -e 1,2p -e 51,53p  -e 102,110p