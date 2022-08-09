
for size in 10 20 30 40 50 500 1000
do
	python3 src/prs/model_learn/train_neural_sat_pref.py --input_file datasets/rand-k-sat/5_${size}_${size}/randkcnf_5_${size}_${size}_0066.cnf  --algo unigen --K 5
done
