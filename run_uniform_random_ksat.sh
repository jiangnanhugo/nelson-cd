#!/usr/bin/zsh
# this script evalutes the uniform sampling from randomly generated KSAT instances

# run the nelson algorithm
algo=$1
K=$2
RV=$3
C=$4
weighted=$5
NUM=100
mkdir -p results/uniform_sample/rand-k-sat/${K}_${RV}_${C}/
for i in {0001..1000}
do

	input_file=datasets/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i.cnf
	output_file=results/uniform_sample/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i
	if [[ "$algo" == "nelson" ]]; then
		echo "nelson $i"
		timeout 60 python3 nelson/sat_uniform_gen.py --samples $NUM --input $input_file --algo nelson --K 5 --weighted $weighted > $output_file.$weighted.nelson.log
	fi

	if [[ "$algo" == "lll" ]]; then
		echo "$algo $i"
		timeout 60 python3 nelson/sat_uniform_gen.py --samples $NUM --input $input_file --algo lll --K 5 --weighted $weighted > $output_file.$weighted.$algo.log
	fi

	if [[ "$algo" == "prs" ]]; then
		echo "$algo $i"
		timeout 60 python3 nelson/sat_uniform_gen.py --samples $NUM --input $input_file --algo prs --K 5  --weighted $weighted> $output_file.$weighted.$algo.log
	fi

	if [[ "$algo" == "numpy" ]]; then
		echo "$algo $i"
		timeout 60 python3 nelson/sat_uniform_gen.py --samples $NUM --input $input_file --algo numpy --K 5 --weighted $weighted > $output_file.$weighted.$algo.log
	fi

	if [[ "$algo" == "unigen" ]]; then
		echo "$algo $i"
		timeout 60 ./uniformSATSampler/unigen --input $input_file --samples $NUM > $output_file.$algo.log
	fi

	if [[ "$algo" == "spur" ]]; then
		echo "$algo $i"
		timeout 60 ./uniformSATSampler/spur -cnf $input_file -s $NUM > $output_file.$algo.log
	fi

	if [[ "$algo" == "cmsgen" ]]; then
		echo "$algo $i"
		timeout 60 ./uniformSATSampler/cmsgen --samples $NUM  $input_file > $output_file.$algo.log
	fi


	if [[ "$algo" == "kus" ]]; then
		echo "$algo $i"
		timeout 60 python3 ./uniformSATSampler/KUS.py --samples $NUM $input_file >$output_file.$algo.log
	fi

	if [[ "$algo" == "quicksampler" ]]; then
		echo "$algo $i"
		timeout 60 ./uniformSATSampler/quicksampler -n $NUM -t 60.0 $input_file >$output_file.$algo.log
	fi


done








