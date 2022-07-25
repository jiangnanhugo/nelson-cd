#!/usr/bin/zsh
# this script evalutes the uniform sampling from randomly generated KSAT instances

# run the nelson algorithm
algo=$1
K=5
NUM=100
timeout=180
for size in {3..20}
do
	RV=${size}00
	C=${size}00
	mkdir -p results/weighted_sample/rand-k-sat/${K}_${RV}_${C}/
	echo "RVs=${RV}, C=${C}"
	for i in {0001..0100}
	do
		input_file=datasets/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i.cnf
		output_file=results/weighted_sample/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i
		if [[ "$algo" == "nelson" || "$algo" == "lll" ||  "$algo" == "prs" || "$algo" == "numpy" || "$algo" == "weightgen" ||  "$algo" == "xor_sampling" || "$algo" == "waps" ]]; then
			echo -ne "$algo $i\r"
			timeout $timeout python3 ./run_weighted_random_ksat.py --samples $NUM --input $input_file --algo $algo --K $K --output $output_file
		fi

	done
done
