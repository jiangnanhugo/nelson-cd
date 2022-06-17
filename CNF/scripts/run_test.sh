for input_file in datasets/simple/*.in
do
    python3 sat.py --input_file $input_file --algo nls
done