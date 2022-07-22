

"""
Solve SAT instance by reading from stdin using the Constructive Lovasz local lemma
or random SAT sampler (trial-and-error) method
"""
import numpy as np
import math
from pysat.formula import CNF
import random
import os
random.seed(10010)
np.random.seed(10010)


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--input', help='read from given cnf file', type=str)
    parser.add_argument('--output', help='read from given cnf file', type=str)
    parser.add_argument('--samples', type=int, default=100,
                        help='number of samples to generate')
    parser.add_argument('--K', type=int, default=3,
                        help="K-SAT")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    cnf_instance = CNF(args.input)
    cnf_instance.to_file(args.input + ".weight")
    prob = np.random.random(cnf_instance.nv)
    with open(args.input + ".weight", "a") as fw:
        for xi in range(cnf_instance.nv):
            fw.write("w {} {} 0\n".format(xi+1, prob[xi]))
            fw.write("w -{} {} 0\n".format(xi+1, 1.0 - prob[xi]))

    if args.algo == 'waps':
        from waps import sampler

        sampler = sampler(cnfFile=args.input + ".weight")
        sampler.compile()
        sampler.parse()
        sampler.annotate()
        samples = sampler.sample(totalSamples=args.samples)
        with open(args.output+".waps.log", 'w')as fw:
            for assignment in samples:
                fw.write(" ".join(assignment)+"\n")
        print("done with waps")
    elif args.algo == 'weightgen':
        kappa = 0.4
        timeout = 72000
        satTimeout = 3000
        epsilon = 0.8
        delta = 0.2
        tilt = 5
        pivotAC = 2 * math.ceil(4.4817 * (1 + 1 / epsilon) * (1 + 1 / epsilon))

        numIterations = int(math.ceil(35 * math.log((3 * 1.0 / delta), 2)))

        pivotUniGen = math.ceil(4.03 * (1 + 1 / kappa) * (1 + 1 / kappa))

        cmd = """./weightedSATSampler/weightgen --samples={} --kappa={} --pivotUniGen={} --maxTotalTime={} \
            --startIteration=0 --maxLoopTime={} --tApproxMC=17 --pivotAC=46 --gaussuntil=400 \
            --verbosity=0 --ratio={} {} {}""".format(args.samples, kappa, pivotUniGen, timeout,
                                                     satTimeout, tilt, args.input + ".weight", args.output+".waps.log")

        os.system(cmd)
    else:
        from argparse import ArgumentParser
        from collections import Counter

        from utils.sat_instance.sat_instance import SAT_Instance
        from nelson.lovasz_sat import *
        from nelson.random_sat import Monte_Carlo_sampler
        import random
        instance = SAT_Instance.from_cnf_file(args.input, args.K)
        device = None
        clause2var, weight, bias = None, None, None
        sampler = None

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
        # Run several times for benchmarking purposes.
        for _ in range(args.samples):
            assignment, count, ti = sampler(instance, clause2var, weight, bias, device=device, prob=prob)
            if len(assignment) > 0:
                result.append(count)
            time_used += ti
        print(Counter(result))
        print("len={}, time={:.4f}s".format(len(result), time_used))