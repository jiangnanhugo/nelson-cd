"""
Solve SAT instance by reading from stdin using the Constructive Lovasz local lemma
or random SAT sampler (trial-and-error) method
"""
import torch
from argparse import ArgumentParser
from collections import Counter


from PRS.utils.sat_instance.sat_instance import SAT_Instance
from lovasz_sat import *
from random_sat import Monte_Carlo_sampler
import random
from statsmodels.stats.gof import chisquare

random.seed(10010)
torch.manual_seed(10010)
np.random.seed(10010)


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--weighted', type=str, help="use weighted sampling?", default='yes')
    parser.add_argument('--input', help='read from given cnf file', type=str)
    parser.add_argument('--K', type=int, default=3,
                        help="K-SAT")
    parser.add_argument('--samples', type=int, default=100,
                        help='number of samples to generate')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    instance = SAT_Instance.from_cnf_file(args.input, args.K)
    device = None
    clause2var, weight, bias = None, None, None
    sampler, prob = None, None
    if args.weighted == 'uniform':
        prob = np.ones(instance.cnf.nv) * 0.5
    elif args.weighted == 'weighted':
        prob = np.random.random(instance.cnf.nv)
    print(prob)

    if args.algo == 'lll':
        sampler = constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'numpy':
        sampler = numpy_conditioned_partial_rejection_sampling_sampler
        clause2var, weight, bias = instance.get_formular_matrix_form()
    elif args.algo == 'nelson':
        sampler = pytorch_neural_lovasz_sampler
        clause2var, weight, bias = instance.get_formular_matrix_form()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clause2var, weight, bias = torch.from_numpy(clause2var).int().to(device), torch.from_numpy(weight).int().to(
            device), torch.from_numpy(bias).int().to(device)
        prob = torch.from_numpy(prob).to(device)

    result = []
    time_used = 0
    all_assignments=[]
    # Run several times for benchmarking purposes.
    for _ in range(args.samples):
        assignment, count, ti = sampler(instance, clause2var, weight, bias, device=device, prob=prob)
        if len(assignment) > 0:
            result.append(count)
            assignment.append(assignment)
        time_used += ti
    print(Counter(result))
    print(Counter(assignment))
    print("len={}, time={:.4f}s".format(len(result), time_used))
