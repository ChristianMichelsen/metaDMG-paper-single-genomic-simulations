# bowtie2-build Mycobacterium_leprae.fa Mycobacterium_leprae.fa

# declare -a damages=("0.014" "0.047" "0.14" "0.30" "0.46" "0.615" "0.93")
declare -a damages=("0" "0001" "001" "014" "047" "14" "30" "46" "615" "93")

# declare -a Nreads=("100000")
declare -a Nreads=("100" "200" "300" "400" "500" "600" "700" "800" "900" "1000" "5000" "10000" "100000" "1000000" "10000000")

# seeds=`seq 1 2`
# seeds=`seq 0 10`
seeds=`seq 0 99`

threads=6
genome=./genome/Mycobacterium_leprae.fa.gz
quality_scores=Test_Examples/AccFreqL150R1.txt

mkdir fastqs
mkdir bams

function simulate_fastq {
    if [ ! -f $fastq.fq ]
    then
        ./ngsngs -i $genome -t $threads -r $Nread -ld Norm,60,10 -s $seed -seq SE -f fq -q1 $quality_scores -b 0.024,0.36,0.$damage,0.0097 -o $fastq
    fi
}

function make_bam_file {
    if [ ! -f $bam ]
    then
        simulate_fastq
        bowtie2 -x $genome -q $fastq.fq --threads $threads --no-unal | samtools view -bS -@ $threads - | samtools sort -n -@ $threads -> $bam
    fi

}

## now loop through the above array
for damage in "${damages[@]}"
do
    ## now loop through the above array
    for Nread in "${Nreads[@]}"
    do
        for seed in $seeds;
        do

            # echo "$damage, $Nread, $seed"
            basename=sim-$damage-$Nread-$seed
            fastq=./fastqs/$basename
            bam=./bams/$basename.bam

            # make_bam_file

        done
    done
done

#%%

# metaDMG config ./bams/*.bam --damage-mode local --bayesian --long-name --parallel-samples 7 --overwrite
# metaDMG compute config.yaml

#%%


# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.93,0.0097 -o test # 30%
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.615,0.0097 -o test # 20%
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.46,0.0097 -o test  # 15%
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.30,0.0097 -o test  # 10%
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.14,0.0097 -o test  # 5%
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.047,0.0097 -o test # 2%
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.014,0.0097 -o test # 1% 0.01127 / 0.00841
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.0000,0.0097 -o test # 0.00676 / 0.00413
# ./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.0001,0.0097 -o test # 0.00680 / 0.00415
./ngsngs -i ./genome/Mycobacterium_leprae.fa.gz -t 15 -r 1000000 -ld Norm,60,10 -s 1 -seq SE -f fq -q1 Test_Examples/AccFreqL150R1.txt -b 0.024,0.36,0.0010,0.0097 -o test # 0.00712 / 0.00440
bowtie2 -x genome/Mycobacterium_leprae.fa.gz -q test.fq --threads 10 --no-unal | samtools view -bS - | samtools sort -n -> test.bam
# # mapDamage -i test.bam -r genome/Mycobacterium_leprae.fa.gz --no-stats -d mapdam --merge-libraries
# metaDMG config test.bam --damage-mode local --config-file config.yaml --overwrite # --bayesian
# metaDMG compute config.yaml
./metaDMG-cpp getdamage --minlength 10 --printlength 50 --threads 8 test.bam