for input_file in datasets/simple/*.in
do
    python3 train_neural_sat_pref.py --input_file $input_file --algo nls
done