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
    parser.add_argument('--file_type', help='cnf', type=str)
    parser.add_argument('--K', help='cnf', type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    instances = []

    if args.file_type == 'cnf':
        instances.append(SAT_Instance.from_cnf_file(args.input_file, args.K))
    else:
        instances.append(SAT_Instance.from_file(args.input_file, K=3))

    if args.algo == 'lll':
        sampler = constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'nls':
        sampler = pytorch_conditioned_partial_rejection_sampling_sampler

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = []
    for inst in instances:
        # Run several times for benchmarking purposes.
        for _ in range(100):
            assignment, count, _ = sampler(inst, device=device, prob=0.5)
            print("the final assignment is:", assignment)
            if len(assignment) > 0:
                result.append(count)
        print(Counter(result))
