
for size in 100 300 500 700 1000
do
	echo "------------------------------------"
	echo $size
	python3 src/lll/model_learn/train_neural_sat_pref.py --input_file datasets/rand-k-sat/5_${size}_${size}/randkcnf_5_${size}_${size}_0066.cnf  --algo nelson --K 5 --sampler_batch_size 150
done
