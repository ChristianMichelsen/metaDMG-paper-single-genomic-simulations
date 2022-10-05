
FILES="./fastq/*.fq"

out=test.txt
echo "HEADER" > $out

for file in $FILES
do
  echo "Processing $file file..."
    name="$(basename -- $file)"
    echo $name >> $out
    # samtools view $file | cut -f1 | grep -o 'mod.*' | sort | uniq -c >> $out
    awk 'NR % 4 == 1' $file | grep -o 'mod.*' | sort | uniq -c >> $out
done


