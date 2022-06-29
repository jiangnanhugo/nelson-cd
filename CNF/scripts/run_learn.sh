#for input_file in datasets/custom/*.cnf
#do
#    echo $input_file
#    python3 train_neural_sat_pref.py --input_file $input_file --algo nls --file_type cnf
#done
echo "5 30 30"
python3 train_neural_sat_pref.py --input_file datasets/custom/5_30_30/randkcnf_5_30_30_0166.cnf --algo nls --K 5
echo "5 50 50"
python3 train_neural_sat_pref.py --input_file datasets/custom/5_50_50/randkcnf_5_50_50_0166.cnf --algo nls --K 5
echo "5 100 100"
python3 train_neural_sat_pref.py --input_file datasets/custom/5_100_100/randkcnf_5_100_100_0166.cnf --algo nls --K 5
echo "5 300 300"
python3 train_neural_sat_pref.py --input_file datasets/custom/5_300_300/randkcnf_5_300_300_0166.cnf --algo nls --K 5
echo "5 500 500"
python3 train_neural_sat_pref.py --input_file datasets/custom/5_500_500/randkcnf_5_500_500_0166.cnf --algo nls --K 5
