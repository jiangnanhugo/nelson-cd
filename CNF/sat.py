"""
Solve SAT instance by reading from stdin using the Constructive Lovasz local lemma
or random SAT sampler (trial-and-error) method
"""
from argparse import ArgumentParser
from collections import Counter

from sat_instance import SAT_Instance
from sampler import *


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--input_file', help='read from given file', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    instances = []
    with open(args.input_file, 'r') as file:
        instances.append(SAT_Instance.from_file(file, K=3))

    
    if args.algo == 'lll':
        sampler =  constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'nls':
        sampler = pyotrch_conditioned_partial_rejection_sampling_sampler
        print("using pyotrch_conditioned_partial_rejection_sampling_sampler")
    
    result = []
    for inst in instances:
        # Run several times for benchmarking purposes.
        for _ in range(10):
            assignment, count = sampler(inst)
            print("the final assignment is:",assignment)
            if len(assignment)>0:
                result.append(count)
        print(Counter(result))