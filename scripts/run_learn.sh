echo "5 40 40"
python3 src/prs/model_learn/train_neural_sat_pref.py --input_file datasets/rand-k-sat/5_40_40/randkcnf_5_40_40_0066.cnf --algo quicksampler --K 5
#python3 src/prs/model_learn/train_neural_sat_pref.py --input_file datasets/rand-k-sat/5_40_40/randkcnf_5_40_40_0066.cnf --algo nelson_batch --K 5
python3 src/prs/model_learn/train_neural_sat_pref.py --input_file /home/jiangnan/PycharmProjects/partial-rejection-sampling/datasets/rand-k-sat/5_10_10/randkcnf_5_10_10_0066.cnf --algo cmsgen --K 5
#echo "5 50 50"
#python3 train_neural_sat_pref.py --input_file datasets/custom/5_50_50/randkcnf_5_50_50_0166.cnf --algo nls --K 5
#echo "5 100 100"
#python3 train_neural_sat_pref.py --input_file datasets/custom/5_100_100/randkcnf_5_100_100_0166.cnf --algo nls --K 5
#echo "5 300 300"
#python3 train_neural_sat_pref.py --input_file datasets/custom/5_300_300/randkcnf_5_300_300_0166.cnf --algo nls --K 5
#echo "5 500 500"
#python3 train_neural_sat_pref.py --input_file datasets/custom/5_500_500/randkcnf_5_500_500_0166.cnf --algo nls --K 5
#
