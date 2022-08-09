
for size in 500 1000
do
	CUDA_VISIBLE_DEVICES=1 python3 src/prs/model_learn/train_neural_sat_pref.py --input_file datasets/rand-k-sat/5_${size}_${size}/randkcnf_5_${size}_${size}_0066.cnf  --algo quicksampler --K 5
done
