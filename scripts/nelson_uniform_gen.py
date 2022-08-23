"""
Solve SAT instance by reading from stdin using the Constructive Lovasz local lemma
or random SAT sampler (trial-and-error) method
"""
from argparse import ArgumentParser
from collections import Counter

from pysat.solvers import Solver
from pysat.formula import CNF
from lll.utils.sat_instance import get_formula_matrix_form
from lll.sampler.nelson.lovasz_sat import *
import random
from statsmodels.stats.gof import chisquare

random.seed(10010)
torch.manual_seed(10010)
np.random.seed(10010)


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--weighted', type=str, help="use weighted sampling?", default='uniform')
    parser.add_argument('--input', help='read from given cnf file', type=str)
    parser.add_argument('--output', help='write to file', type=str)
    parser.add_argument('--K', type=int, default=3, help="K-SAT")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size", required=False)
    parser.add_argument('--samples', type=int, default=100, help='number of samples to generate')
    return parser.parse_args()


def lll(samples, prob):
    all_assignments = []
    time_used = 0
    result = []
    for _ in range(samples):
        batched_assignment, count, ti = constructive_lovasz_local_lemma_sampler(cnf_instance, prob=prob)
        if len(batched_assignment) > 0:
            result.append(count)
            all_assignments.append("".join([str(xx) for xx in batched_assignment.tolist()]))
        time_used += ti
    return all_assignments, time_used, result


def prs(samples, prob):
    all_assignments = []
    time_used = 0
    result = []
    for _ in range(samples):
        batched_assignment, count, ti = partial_rejection_sampling_sampler(cnf_instance, prob=prob)
        if len(batched_assignment) > 0:
            result.append(count)
            all_assignments.append("".join([str(xx) for xx in batched_assignment.tolist()]))
        time_used += ti
    return all_assignments, time_used, result


def numpy_nelson(samples, prob, K=5):
    clause2var, weight, bias = get_formula_matrix_form(cnf_instance, K)
    all_assignments = []
    time_used = 0
    result = []
    # Run several times for benchmarking purposes.
    for _ in range(samples):
        batched_assignment, count, ti = numpy_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias,
                                                                    prob=prob)
        if len(batched_assignment) > 0:
            result.append(count)
            all_assignments.append("".join([str(xx) for xx in batched_assignment.tolist()]))
        time_used += ti
    return all_assignments, time_used, result


def numpy_batch_nelson(samples, prob, batch_size, K=5):
    clause2var, weight, bias = get_formula_matrix_form(cnf_instance, K)
    all_assignments = []
    time_used = 0
    result = []
    # Run several times for benchmarking purposes.
    for _ in range(samples // batch_size):
        batched_assignment, count, ti = numpy_batch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias,
                                                                          prob=prob, batch_size=batch_size)
        if len(batched_assignment) > 0:
            result.append(count)
            for x in batched_assignment.tolist():
                all_assignments.append("".join([str(xx) for xx in x]))
        time_used += ti
    if samples % batch_size != 0:
        batched_assignment, count, ti = numpy_batch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias, prob=prob,
                                                                          batch_size=args.samples % batch_size)
        if len(batched_assignment) > 0:
            result.append(count)
            for x in batched_assignment.tolist():
                all_assignments.append("".join([str(xx) for xx in x]))
            time_used += ti
    return all_assignments, time_used, result


def pytorch_nelson(samples, prob, K=5):
    clause2var, weight, bias = get_formula_matrix_form(cnf_instance, K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clause2var, weight, bias = torch.from_numpy(clause2var).int().to(device), torch.from_numpy(weight).int().to(
        device), torch.from_numpy(bias).int().to(device)
    prob = torch.from_numpy(prob).to(device)
    all_assignments = []
    time_used = 0
    result = []
    # Run several times for benchmarking purposes.
    for _ in range(samples):
        batched_assignment, count, ti = pytorch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias,
                                                                      device=device, prob=prob)
        if len(batched_assignment) > 0:
            result.append(count)
            all_assignments.append("".join([str(xx) for xx in batched_assignment.tolist()]))
        time_used += ti

    return all_assignments, time_used, result


def pytorch_batch_nelson(samples, prob, batch_size, K=5):
    clause2var, weight, bias = get_formula_matrix_form(cnf_instance, K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clause2var, weight, bias = torch.from_numpy(clause2var).int().to(device), torch.from_numpy(weight).int().to(
        device), torch.from_numpy(bias).int().to(device)
    clause2var = torch.transpose(clause2var, 0, 1)
    clause2var = clause2var.reshape(1, *clause2var.size())
    weight = weight.reshape(1, *weight.size())
    bias = bias.reshape(1, *bias.size())
    prob = torch.from_numpy(prob).to(device)
    all_assignments = []
    time_used = 0
    result = []
    # Run several times for benchmarking purposes.
    for _ in range(samples // batch_size):

        batched_assignment, count, ti = pytorch_batch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias,
                                                                            device=device, prob=prob, batch_size=batch_size)
        if len(batched_assignment) > 0:
            result.append(count)
            for x in batched_assignment.tolist():
                all_assignments.append("".join([str(xx) for xx in x]))
        time_used += ti
    if samples % batch_size != 0:
        batched_assignment, count, ti = pytorch_batch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias, device=device, prob=prob,
                                                                            batch_size=args.samples % batch_size)
        if len(batched_assignment) > 0:
            result.append(count)
            for x in batched_assignment.tolist():
                all_assignments.append("".join([str(xx) for xx in x]))
            time_used += ti
    return all_assignments, time_used, result


if __name__ == "__main__":
    args = get_arguments()
    cnf_instance = CNF(args.input, args.K)

    solver = Solver(bootstrap_with=cnf_instance.clauses)
    batch_size = args.batch_size

    prob = np.ones(cnf_instance.nv) * 0.5
    if args.weighted == 'weighted':
        prob = np.random.random(cnf_instance.nv)

    if args.algo == 'lll':
        all_assignments, time_used, result = lll(samples=args.samples, prob=prob)
    elif args.algo == 'lll':
        all_assignments, time_used, result = prs(samples=args.samples, prob=prob)
    elif args.algo == 'numpy':
        all_assignments, time_used, result = numpy_nelson(samples=args.samples, prob=prob)
    elif args.algo == 'numpy_batch':
        all_assignments, time_used, result = numpy_batch_nelson(samples=args.samples, prob=prob, batch_size=args.batch_size)
    elif args.algo == 'nelson':
        all_assignments, time_used, result = pytorch_nelson(samples=args.samples, prob=prob)
    elif args.algo == 'nelson_batch':
        all_assignments, time_used, result = pytorch_batch_nelson(samples=args.samples, prob=prob, batch_size=args.batch_size)

    freq_dict = Counter(all_assignments)
    # print(freq_dict)
    cnt_valid = 0
    dists = []
    for k in freq_dict:
        assignment = []
        for idx, x in enumerate(k):
            if x == '0':
                assignment.append(-idx - 1)
            else:
                assignment.append(idx + 1)
        if solver.solve(assignment) == True:
            cnt_valid += freq_dict[k]
        dists.append(freq_dict[k])

    dists = []
    freq = dict(Counter(all_assignments))
    for k in freq:
        dists.append(freq[k])
    result = chisquare(dists)
    print("freq={}".format(len(dists)))
    print("chisquare: {:.2f}".format(result[0]))

    print("valid {}".format(cnt_valid / args.samples))
    print(len(dists), result[0])
    print("sample time {:.6f}".format(time_used))
    with open(args.output, 'w') as fw:
        fw.write("freq={}\n".format(len(dists)))
        fw.write("valid {:.4f}\n".format(100 * cnt_valid / args.samples))
        fw.write("chisquare {:.2f}\n".format(result[0]))
        fw.write("sample time {:.6f}".format(time_used))
