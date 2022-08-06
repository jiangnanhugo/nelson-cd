#!/usr/bin/zsh
# this script evalutes the uniform sampling from randomly generated KSAT instances

# run the nelson algorithm
algo=$1
K=5
NUM=$2
size=$3
batch_size=$4
timeout=7200
RV=$size
C=$size
mkdir -p results/uniform_sample/rand-k-sat/${K}_${RV}_${C}/
for i in {0001..0020}
do
    input_file=datasets/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i.cnf
    output_file=results/uniform_sample/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i
    if [[ "$algo" == "nelson" || "$algo" == "lll" || "$algo" == "prs" || "$algo" == "numpy" ]]; then
        echo "$i"
        timeout $timeout python3 ./scripts/nelson_uniform_gen.py --samples $NUM --input $input_file --algo $algo --K 5 --weighted uniform  --output $output_file.$algo.log
    fi

    if [[ "$algo" == "nelson_batch" || "$algo" == "numpy_batch" ]]; then
        echo "$i"
        timeout $timeout python3 ./scripts/nelson_uniform_gen.py --samples $NUM --input $input_file --algo $algo --K 5 --weighted uniform --batch_size $batch_size --output $output_file.$algo.log
    fi

    if [[ "$algo" == "unigen" || "$algo" == "kus"  || "$algo" == "cmsgen" || "$algo" == "quicksampler" ]]; then
        echo "$algo $i"
        python3 ./scripts/run_uniform_random_ksat.py  --input $input_file --samples $NUM --algo $algo --output $output_file
    fi
done
