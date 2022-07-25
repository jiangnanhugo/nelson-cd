import numpy as np
import math
from utils.sat_instance.sat_instance import SAT_Instance
from pysat.formula import CNF
import random
from argparse import ArgumentParser
import time
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
    instance = SAT_Instance.from_cnf_file(args.input, args.K)

    instance.cnf.to_file(args.input + ".weight")
    prob = np.random.random(instance.cnf.nv)

    with open(args.input + ".weight", "a") as fw:
        for xi in range(instance.cnf.nv):
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
        cmd = """./sampler/weightedSATSampler/weightgen --samples={} --kappa={} --pivotUniGen={} --maxTotalTime={} \
            --startIteration=0 --maxLoopTime={} --tApproxMC=17 --pivotAC=46 --gaussuntil=400 \
            --verbosity=0 --ratio={} {} {}""".format(args.samples, kappa, pivotUniGen, timeout,
                                                     satTimeout, tilt, args.input + ".weight", args.output+".weightgen.log")

        os.system(cmd)
        sampled_time = time.time() - st
        with open(args.output+".weightgen.log", 'a')as fw:
            fw.write("sample time: {:.6f}".format(sampled_time))
        print("sample time: {:.6f}".format(sampled_time))
    elif args.algo == 'xor_sampling':
        from sampler.xor_sampling.xor_sampler import XOR_Sampling
        from utils.cnf2uai import cnf_to_uai
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


        from sampler.nelson.lovasz_sat import *
        from sampler.nelson.random_sat import Monte_Carlo_sampler
        import random

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
        assignments = []
        time_used = 0
        # Run several times for benchmarking purposes.
        for _ in range(args.samples):
            assignment, count, ti = sampler(instance, clause2var, weight, bias, device=device, prob=prob)
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

