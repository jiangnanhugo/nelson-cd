
for size in 700
do
	echo "------------------------------------"
	echo $size
	CUDA_VISIBLE_DEVICES=0 python3 src/lll/model_learn/train_neural_sat_pref.py --input_file datasets/rand-k-sat/5_${size}_${size}/randkcnf_5_${size}_${size}_0066.cnf  --algo gibbs_sampling --K 5
done
