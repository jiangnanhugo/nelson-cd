"""
Solve SAT instance by reading from stdin using the Constructive Lovasz local lemma
or random SAT sampler (trial-and-error) method
"""
from argparse import ArgumentParser
from collections import Counter

from sat_instance import SAT_Instance
from sampler import constructive_lovasz_local_lemma_sampler, partial_rejection_sampling_sampler, Monte_Carlo_sampler


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output.', action='store_true')
    parser.add_argument('-a', '--all', help='output all possible solutions.', action='store_true')
    parser.add_argument('-b', '--brief', help='brief output: only outputs variables assigned true.', action='store_true')
    parser.add_argument('--starting_with', help='only output variables with names starting with the given string.', default='')

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--input_file', help='read from given file', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    instances = []
    with open(args.input_file, 'r') as file:
        instances.append(SAT_Instance.from_file(file))

    
    if args.algo == 'lll':
        sampler =  constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = partial_rejection_sampling_sampler
    result = []
    for inst in instances:
        # Run several times for benchmarking purposes.
        for _ in range(10000):
            assignment, count = sampler(inst)
            if assignment != None:
                result.append(count)
        print(Counter(result))

        # if args.verbose and assignment != None:
        #     print('Found satisfying assignment:')
        #     print(inst.assignment_to_string(assignment,
        #                                         brief=args.brief,
        #                                         starting_with=args.starting_with))
        # elif args.verbose:
        #     print('No satisfying assignment exists.', file=stderr)