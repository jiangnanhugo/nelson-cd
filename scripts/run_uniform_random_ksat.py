import numpy as np
from pysat.formula import CNF
from pysat.solvers import Solver
import random
from collections import Counter
from argparse import ArgumentParser
import time
import os

from statsmodels.stats.gof import chisquare

s = Solver()
random.seed(10010)
np.random.seed(10010)


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--input', help='read from given cnf file', type=str)
    parser.add_argument('--output', help='read from given cnf file', type=str, default="sample.output.log")
    parser.add_argument('--samples', type=int, default=100,
                        help='number of samples to generate')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    formula = CNF(args.input)
    solver = Solver(bootstrap_with=formula.clauses)
    cnt_valid = 0
    dists = []
    sampled_time = 0
    if args.algo == 'kus':
        tmpfile = "/tmp/randksat.kus.txt"
        cmd = "python3  ./src/lll/sampler/uniformSATSampler/KUS.py --samples {} --outputfile {} {} >/tmp/tmp.log".format(args.samples,
                                                                                                                         tmpfile,
                                                                                                                         args.input)
        st = time.time()
        os.system(cmd)
        sampled_time = time.time() - st
        with open(tmpfile, 'r') as fr:
            lines = []
            for x in fr.read().split("\n"):
                if len(x) > 1:
                    lines.append(x)
        freq_dict = Counter(lines)

        for k in freq_dict:
            assignment = [0, ] * len(k.strip().split(" "))
            for x in k.strip().split(" "):
                assignment[abs(int(x)) - 1] = int(x)
            if solver.solve(assignment) == True:
                cnt_valid += freq_dict[k]
            dists.append(freq_dict[k])
        result = chisquare(dists)
    elif args.algo == 'cmsgen':
        tmpfile = "/tmp/randksat.cmsgen.txt"
        cmd = "./src/lll/sampler/uniformSATSampler/cmsgen --samples {} --samplefile {} {} >/tmp/tmp.log".format(args.samples,
                                                                                                                tmpfile,
                                                                                                                args.input)
        st = time.time()
        os.system(cmd)
        sampled_time = time.time() - st

        with open(tmpfile, 'r') as fr:
            lines = []
            for x in fr.read().split("\n"):
                if len(x) > 1:
                    lines.append(x)
        freq_dict = Counter(lines)

        for k in freq_dict:
            assignment = [int(x) for x in k.split(" ")[:-1]]
            if solver.solve(assignment) == True:
                cnt_valid += freq_dict[k]
            dists.append(freq_dict[k])
        result = chisquare(dists)

    elif args.algo == 'unigen':
        tmpfile = '/tmp/unigen.txt'
        cmd = """./src/lll/sampler/uniformSATSampler/unigen --input {} --samples {} --sampleout {} > /tmp/tmp.txt""".format(args.input,
                                                                                                                            args.samples,
                                                                                                                            tmpfile)
        st = time.time()

        os.system(cmd)
        sampled_time = time.time() - st
        with open(tmpfile, 'r') as fr:
            lines = []
            for x in fr.read().split("\n"):
                if len(x) > 1:
                    lines.append(x)
                if len(lines) >= args.samples:
                    break
        freq_dict = Counter(lines)

        for k in freq_dict:
            assignment = [int(x) for x in k.split(" ")[:-1]]
            if solver.solve(assignment) == True:
                cnt_valid += freq_dict[k]
            dists.append(freq_dict[k])
        result = chisquare(dists)

    elif args.algo == 'quicksampler':
        print(args.input)
        cmd = """./src/lll/sampler/uniformSATSampler/quicksampler -n {} -t 180.0 {} >/tmp/tmp.log""".format(args.samples, args.input)
        st = time.time()

        os.system(cmd)
        sampled_time = time.time() - st
        with open(args.input + '.samples', 'r') as fr:
            lines = []
            idx = 0
            for li in fr.read().split("\n"):
                instance = li.strip().split(" ")[-1].strip()
                if len(instance) <= 1:
                    continue
                if idx >= args.samples:
                    break
                lines.append(instance)
                idx += 1
        os.remove(args.input + '.samples')

        freq_dict = Counter(lines)
        cnt = 0
        for k in freq_dict:
            assignment = []
            # print(len(k))
            for idx, x in enumerate(k):
                if x == '0':
                    assignment.append(-idx - 1)
                else:
                    assignment.append(idx + 1)
            if solver.solve(assignment):
                cnt_valid += freq_dict[k]
                dists.append(freq_dict[k])
            else:
                cnt += 1
        print(cnt, len(lines))
    elif args.algo == 'gibbs_sampling':
        from lll.sampler.gibbs_sampler.gibbs_mrf import Gibbs_Sampling
        from lll.utils.cnf2uai import cnf_to_uai

        st = time.time()
        prob = np.ones(formula.nv)
        cnf_to_uai(formula, prob, args.input + ".uniform.gibbs.uai")
        returned_samples = Gibbs_Sampling(args.input + ".uniform.gibbs.uai", args.samples)

        sampled_time = time.time() - st
        lines = []
        for xx in returned_samples:
            lines.append("".join([str(xxx) for xxx in xx]))

        freq_dict = Counter(lines)

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

    elif args.algo == 'xor_sampling':
        from lll.sampler.xor_sampling.xor_sampler import XOR_Sampling
        from lll.utils.cnf2uai import cnf_to_uai

        prob = np.ones(formula.nv)
        cnf_to_uai(formula, prob, args.input + ".uniform.xor.uai")
        st = time.time()
        returned_samples = XOR_Sampling(args.input + ".uniform.xor.uai", args.samples)

        sampled_time = time.time() - st
        with open(args.output + ".xor_sampling.log", 'a') as fw:
            for xx in returned_samples:
                fw.write("{}\n".format(" ".join([str(xxx) for xxx in xx])))
            fw.write("sample time: {:.6f}".format(sampled_time))

    result = chisquare(dists)
    print("valid {}".format(cnt_valid / args.samples))
    print(len(dists), result[0])
    print("sample time {:.6f}".format(sampled_time))
    with open(args.output + "." + args.algo + ".log", 'w') as fw:
        fw.write("freq={}\n".format(len(dists)))
        fw.write("valid {:.4f}\n".format(100 * cnt_valid / args.samples))
        fw.write("chisquare {}\n".format(result[0]))
        fw.write("sample time {:.6f}".format(sampled_time))
