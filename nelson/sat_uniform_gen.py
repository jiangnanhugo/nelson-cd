"""
Solve SAT instance by reading from stdin using the Constructive Lovasz local lemma
or random SAT sampler (trial-and-error) method
"""
import torch
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm

from sat_instance import SAT_Instance
from lovasz_sat import *
from random_sat import Monte_Carlo_sampler


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--input', help='read from given cnf file', type=str)
    parser.add_argument('--K', type=int, default=3,
                        help="K-SAT")
    parser.add_argument('--samples', type=int, default=100,
                        help='number of samples to generate')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    instance = SAT_Instance.from_cnf_file(args.input, args.K)
    device=None
    if args.algo == 'lll':
        sampler = constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'numpy':
        sampler = numpy_conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'nelson':
        sampler = pytorch_neural_lovasz_sampler

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = []
    time_used = 0
    # Run several times for benchmarking purposes.
    for _ in tqdm(range(args.samples)):
        assignment, count, ti = sampler(instance, device=device, prob=0.5)
        if len(assignment) > 0:
            result.append(count)
        time_used += ti
    print(Counter(result))
    print("len={}, time={:.4f}s".format(len(result), time_used))
