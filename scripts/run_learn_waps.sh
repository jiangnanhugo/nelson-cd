
for size in 100
do
	echo "------------------------------------"
	echo $size
	CUDA_VISIBLE_DEVICES=1 python3 src/lll/model_learn/train_neural_sat_pref.py --input_file datasets/rand-k-sat/5_${size}_${size}/randkcnf_5_${size}_${size}_0066.cnf  --algo waps --K 5
done
