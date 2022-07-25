#!/usr/bin/zsh
# this script evalutes the uniform sampling from randomly generated KSAT instances

# run the nelson algorithm
algo=$1
K=$2
RV=$3
C=$4
NUM=10000
timeout=180
mkdir -p results/uniform_sample/rand-k-sat/${K}_${RV}_${C}/
for i in {0001..0100}
do
	input_file=datasets/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i.cnf
	output_file=results/uniform_sample/rand-k-sat/${K}_${RV}_${C}/randkcnf_${K}_${RV}_${C}_$i
	if [[ "$algo" == "nelson" || "$algo" == "lll" || "$algo" == "prs" || "$algo" == "numpy"  ]]; then
		echo "nelson $i"
		timeout $timeout python3 sampler/nelson/nelson_gen.py --samples $NUM --input $input_file --algo $algo --K 5 --weighted uniform > $output_file.uniform.$algo.log
	fi

	if [[ "$algo" == "unigen" ]]; then
		echo "$algo $i"
		timeout $timeout ./uniformSATSampler/unigen --input $input_file --samples $NUM > $output_file.$algo.log
	fi

	if [[ "$algo" == "spur" ]]; then
		echo "$algo $i"
		timeout $timeout ./uniformSATSampler/spur -cnf $input_file -s $NUM > $output_file.$algo.log
	fi

	if [[ "$algo" == "cmsgen" ]]; then
		echo "$algo $i"
		timeout $timeout ./uniformSATSampler/cmsgen --samples $NUM  $input_file > $output_file.$algo.log
	fi


	if [[ "$algo" == "kus" ]]; then
		echo "$algo $i"
		timeout $timeout python3 ./uniformSATSampler/KUS.py --samples $NUM $input_file >$output_file.$algo.log
	fi

	if [[ "$algo" == "quicksampler" ]]; then
		echo "$algo $i"
		timeout $timeout ./uniformSATSampler/quicksampler -n $NUM -t 60.0 $input_file >$output_file.$algo.log
	fi
done








