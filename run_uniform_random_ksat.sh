# this script evalutes the uniform sampling from randomly generated KSAT instances

# run the nelson algorithm

input_file=/home/jiangnan/PycharmProjects/partial-rejection-sampling/datasets/rand-k-sat/5_300_300/randkcnf_5_300_300_0002.cnf
python3 nelson/sat_uniform_gen.py --samples 100 --input $input_file --algo nelson --K 5


./uniformSATSampler/unigen --input $input_file --samples 100

./uniformSATSampler/spur/spur -s 100 -cnf $input_file


