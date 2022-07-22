#!/usr/bin/zsh
# this script evalutes the uniform sampling from randomly generated KSAT instances

# run the nelson algorithm
algo=$1
K=$2
RV=$3
C=$4
NUM=100
timeout=7200
mkdir -p results/weighted_sample/rand-k-sat/${K}_${RV}_${C}/
for i in {0001..0010}
do

	input_file=datasets/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i.cnf
	output_file=results/weighted_sample/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i
	if [[ "$algo" == "nelson" ]]; then
		echo "nelson $i"
		timeout $timeout python3 ./run_weighted_random_ksat.py --samples $NUM --input $input_file --algo nelson --K 5 --output $output_file
	fi

	if [[ "$algo" == "lll" ]]; then
		echo "$algo $i"
		timeout $timeout python3 ./run_weighted_random_ksat.py --samples $NUM --input $input_file --algo lll --K 5 --output $output_file
	fi

	if [[ "$algo" == "prs" ]]; then
		echo "$algo $i"
		timeout $timeout python3 ./run_weighted_random_ksat.py --samples $NUM --input $input_file --algo prs --K 5  --output $output_file
	fi

	if [[ "$algo" == "numpy" ]]; then
		echo "$algo $i"
		timeout $timeout python3 ./run_weighted_random_ksat.py --samples $NUM --input $input_file --algo numpy --K 5 --output $output_file
	fi

	if [[ "$algo" == "waps" ]]; then
		echo "$algo $i"
		timeout $timeout python3 ./run_weighted_random_ksat.py --samples $NUM --input $input_file --algo waps --output $output_file
	fi

	if [[ "$algo" == "weightgen" ]]; then
		echo "$algo $i"
		timeout $timeout python3 ./run_weighted_random_ksat.py --samples $NUM --input $input_file --algo weightgen --output $output_file
	fi
done








