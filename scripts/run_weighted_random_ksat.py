import numpy as np
import math
from lll.utils.sat_instance import get_formula_matrix_form
import random
from argparse import ArgumentParser
import time
import os
random.seed(10010)
np.random.seed(10010)
from pysat.formula import CNF

def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--input_file', help='read from given cnf file', type=str)
    parser.add_argument('--output', help='read from given cnf file', type=str)
    parser.add_argument('--samples', type=int, default=100,
                        help='number of samples to generate')
    parser.add_argument('--K', type=int, default=3,
                        help="K-SAT")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    cnf_instance = CNF(args.input_file, K=args.K)

    cnf_instance.to_file(args.input + ".weight")
    prob = np.random.random(cnf_instance.nv)

    with open(args.input + ".weight", "a") as fw:
        for xi in range(cnf_instance.nv):
            fw.write("w {} {} 0\n".format(xi+1, prob[xi]))
            fw.write("w -{} {} 0\n".format(xi+1, 1.0 - prob[xi]))

    if args.algo == 'waps':
        from waps import sampler
        st = time.time()
        sampler = sampler(cnfFile=args.input + ".weight")
        sampler.compile()
        sampler.parse()
        sampler.annotate()
        compile_time = time.time() - st
        st = time.time()
        samples = sampler.sample(totalSamples=args.samples)
        sample_time = time.time() - st
        with open(args.output+".waps.log", 'w')as fw:
            for assignment in samples:
                fw.write(" ".join(assignment)+"\n")
            fw.write("compile: {:.6f}\nsample: {:.6f}".format(compile_time, sample_time))
        print("compile: {:.6f}\nsample time: {:.6f}".format(compile_time, sample_time))
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
        st = time.time()
        cmd = """./src/lll/sampler/weightedSATSampler/weightgen --samples={} --kappa={} --pivotUniGen={} --maxTotalTime={} \
            --startIteration=0 --maxLoopTime={} --tApproxMC=17 --pivotAC=46 --gaussuntil=400 \
            --verbosity=0 --ratio={} {} {}""".format(args.samples, kappa, pivotUniGen, timeout,
                                                     satTimeout, tilt, args.input + ".weight", args.output+".weightgen.log")

        os.system(cmd)
        sampled_time = time.time() - st
        with open(args.output+".weightgen.log", 'a')as fw:
            fw.write("sample time: {:.6f}".format(sampled_time))
        print("sample time: {:.6f}".format(sampled_time))
    elif args.algo == 'xor_sampling':
        from lll.sampler.xor_sampling.xor_sampler import XOR_Sampling
        from lll.utils.cnf2uai import cnf_to_uai
        cnf_to_uai(cnf_instance, prob, args.input + ".weight.uai")
        st = time.time()
        returned_samples = XOR_Sampling(args.input + ".weight.uai", args.samples)

        sampled_time = time.time() - st
        with open(args.output + ".xor_sampling.log", 'a') as fw:
            for xx in returned_samples:
                fw.write("{}\n".format(" ".join([str(xxx) for xxx in xx])))
            fw.write("sample time: {:.6f}".format(sampled_time))
    else:
        from collections import Counter

        from lll.sampler.nelson.lovasz_sat import pytorch_neural_lovasz_sampler, constructive_lovasz_local_lemma_sampler, \
             pytorch_batch_neural_lovasz_sampler, numpy_neural_lovasz_sampler
        from lll.sampler.nelson.random_sat import Monte_Carlo_sampler
        import random
        import torch

        device = None
        V, W, b = None, None, None
        sampler = None

        if args.algo == 'lll':
            sampler = constructive_lovasz_local_lemma_sampler
        elif args.algo == 'mc':
            sampler = Monte_Carlo_sampler
        # elif args.algo == 'lll':
        #     sampler = partial_rejection_sampling_sampler
        elif args.algo == 'numpy':
            sampler = numpy_neural_lovasz_sampler
            V, W, b = get_formula_matrix_form(cnf_instance, args.K)
        elif args.algo == 'nelson':
            sampler = pytorch_neural_lovasz_sampler
            V, W, b = get_formula_matrix_form(cnf_instance, args.K)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            V, W, b = torch.from_numpy(V).int().to(device), torch.from_numpy(W).int().to(
                device), torch.from_numpy(b).int().to(device)
            prob = torch.from_numpy(prob).to(device)

        result = []
        assignments = []
        time_used = 0
        # Run several times for benchmarking purposes.
        for _ in range(args.samples):
            assignment, count, ti = sampler(cnf_instance, V, W, b, device=device, prob=prob)
            if len(assignment) > 0:
                result.append(count)
                assignments.append(assignment)
            time_used += ti

        with open(args.output + "."+args.algo+".log", 'w') as fw:
            for xx in assignments:
                fw.write("{}\n".format(" ".join([str(xxx) for xxx in xx])))
            fw.write(str(dict(Counter(result)))+"\n")
            fw.write("len={}\nsample time {:.6f}".format(len(result), time_used))
        # print("sample time: {:.4f}".format(sampled_time))

