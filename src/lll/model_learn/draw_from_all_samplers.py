import sys
import time
import torch
import math
import os
import numpy as np
from collections import Counter
from waps import sampler as waps_sampler

from pysat.formula import CNF
from pysat.solvers import Solver

from lll.sampler.nelson.lovasz_sat import pytorch_neural_lovasz_sampler, constructive_lovasz_local_lemma_sampler, \
    partial_rejection_sampling_sampler, pytorch_batch_neural_lovasz_sampler, numpy_neural_lovasz_sampler
from lll.sampler.nelson.random_sat import Monte_Carlo_sampler
from lll.sampler.xor_sampling.xor_sampler import XOR_Sampling
from lll.sampler.gibbs_sampler.gibbs_mrf import Gibbs_Sampling
from lll.utils.cnf2uai import cnf_to_uai

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_from_xor_sampling(num_samples, input_file, cnf_instance, prob):
    cnf_to_uai(cnf_instance, prob, input_file + ".weight.uai")
    returned_samples = XOR_Sampling(input_file + ".weight.uai", num_samples)

    sampled_assignments = []
    for sampled_assig in returned_samples:
        sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments


def draw_from_gibbs_sampling(num_samples, input_file, cnf_instance, prob):
    cnf_to_uai(cnf_instance, prob, input_file + ".weight.uai")
    returned_samples = Gibbs_Sampling(input_file + ".weight.uai", num_samples)

    sampled_assignments = []
    for sampled_assig in returned_samples:
        sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments


def draw_from_weightgen(num_samples, input_file, cnf_instance, prob, device='cuda'):
    cnf_instance.to_file(input_file + ".weight")
    with open(input_file + ".weight", "a") as fw:
        for xi in range(cnf_instance.nv):
            fw.write("w {} {} 0\n".format(xi + 1, 1.0 - prob[xi]))
            fw.write("w -{} {} 0\n".format(xi + 1, prob[xi]))
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
    tmp_file = "/tmp/randksat.weightgen.log"
    cmd = """./src/lll/sampler/weightedSATSampler/weightgen --samples={} --kappa={} --pivotUniGen={} --maxTotalTime={} \
                --startIteration=0 --maxLoopTime={} --tApproxMC=17 --pivotAC=46 --gaussuntil=400 \
                --verbosity=0 --ratio={} {} {}""".format(num_samples, kappa, pivotUniGen, timeout,
                                                         satTimeout, tilt, input_file + ".weight", tmp_file)

    os.system(cmd)
    sampled_assignments = []
    with open(tmp_file, 'r') as fr:
        for li in fr:
            if len(li) <= 1:
                continue
            one_sol = li.strip().split(" ")[1:-1]
            sampled_assig = [1 if int(x) > 0 else 0 for x in one_sol]
            sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments


def draw_from_prs_series(algo, Fi, clause2var, weight, bias, prob, num_samples, **kwargs):
    if algo == 'lll':
        sampler = constructive_lovasz_local_lemma_sampler
    elif algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif algo == 'lll':
        sampler = partial_rejection_sampling_sampler
    elif algo == 'numpy':
        clause2var, weight, bias = Fi.get_formular_matrix_form()

        samples = []
        for _ in range(num_samples):
            assignment, count, ti = numpy_neural_lovasz_sampler(Fi, clause2var, weight, bias, prob=prob)
            if len(assignment) > 0:
                samples.append(torch.from_numpy(assignment))
    elif algo == 'nelson':
        samples = []
        while len(samples) <= num_samples:
            assignment, count, _ = pytorch_neural_lovasz_sampler(Fi, clause2var, weight, bias, device=device, prob=prob)
            if len(assignment) > 0:
                samples.append(assignment.reshape(1, -1))

    elif algo == 'nelson_batch':
        batch_size = kwargs['sampler_batch_size']
        clause2var = torch.transpose(clause2var, 0, 1)
        clause2var = clause2var.reshape(1, *clause2var.size())
        weight = weight.reshape(1, *weight.size())
        bias = bias.reshape(1, *bias.size())
        samples = []
        for _ in range(num_samples // batch_size):

            batched_assignment, _, _ = pytorch_batch_neural_lovasz_sampler(Fi, clause2var, weight, bias,
                                                                           device=device, prob=prob, batch_size=batch_size)
            if len(batched_assignment) > 0:
                samples += batched_assignment
        if num_samples % batch_size != 0:
            batched_assignment, count, ti = pytorch_batch_neural_lovasz_sampler(Fi, clause2var, weight, bias, device=device,
                                                                                prob=prob,
                                                                                batch_size=num_samples % batch_size)
            if len(batched_assignment) > 0:
                samples += batched_assignment
    return samples


def draw_from_waps(num_samples, input_file, cnf_instance, prob):
    cnf_instance.to_file(input_file + ".weight")
    with open(input_file + ".weight", "a") as fw:
        for xi in range(cnf_instance.nv):
            fw.write("w {} {} 0\n".format(xi + 1, 1 - prob[xi]))
            fw.write("w -{} {} 0\n".format(xi + 1, prob[xi]))
    sampler = waps_sampler(cnfFile=input_file + ".weight")
    sampler.compile()
    sampler.parse()
    sampler.annotate()
    samples = sampler.sample(totalSamples=num_samples)
    sampled_assignments = []
    for li in samples:
        if len(li) <= 1:
            continue
        one_sol = li.strip().split(" ")
        sampled_assig = [1 if int(x) > 0 else 0 for x in one_sol]
        sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments


def draw_from_quicksampler(num_samples, input_file):
    sampled_assignments = []
    formula = CNF(input_file)
    solver = Solver(bootstrap_with=formula.clauses)
    iter = 0
    while len(sampled_assignments) < num_samples:
        iter += 1
        cmd = """./src/lll/sampler/uniformSATSampler/quicksampler -n {} -t 180.0 {} >/tmp/tmp.log""".format(num_samples, input_file)

        os.system(cmd)
        print(iter, len(sampled_assignments), end="\r")
        sys.stdout.flush()
        lines = []
        with open(input_file + '.samples', 'r') as fr:
            for li in fr.read().split("\n"):
                one_sol = li.strip().split(" ")[-1].strip()
                if len(one_sol) <= 1:
                    continue
                lines.append(one_sol)
        os.remove(input_file + '.samples')
        cnt = 0
        for k in lines:
            assignment = []
            for idx, x in enumerate(k):
                if x == '0':
                    assignment.append(-idx - 1)
                else:
                    assignment.append(idx + 1)
            if solver.solve(assignment):
                sampled_assignments.append([1 if int(x) > 0 else 0 for x in k])
            else:
                cnt += 1

    sampled_assignments = [torch.from_numpy(np.array(sampled_assignments[idx])).to(device).reshape(1, -1) for idx in range(num_samples)]

    return sampled_assignments


def draw_from_kus(num_samples, input_file):
    tmpfile = "/tmp/randksat.kus.txt"
    cmd = "python3  ./src/lll/sampler/uniformSATSampler/KUS.py --samples {} --outputfile {} {} >/tmp/tmp.log".format(num_samples,
                                                                                                                     tmpfile,
                                                                                                                     input_file)

    os.system(cmd)
    sampled_assignments = []
    with open(tmpfile, 'r') as fr:
        for li in fr.read().split("\n"):
            one_sol = li.strip().split(" ")

            if len(li) <= 1:
                continue
            sampled_assig = [0, ] * len(one_sol)
            for x in one_sol:
                if int(x) > 0:
                    sampled_assig[abs(int(x)) - 1] = 1
            sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments


def draw_from_cmsgen(num_samples, input_file):
    tmpfile = "/tmp/randksat.cmsgen.txt"
    cmd = "./src/lll/sampler/uniformSATSampler/cmsgen --samples {} --samplefile {} {} >/tmp/tmp.log".format(num_samples,
                                                                                                            tmpfile,
                                                                                                            input_file)
    os.system(cmd)
    sampled_assignments = []
    with open(tmpfile, 'r') as fr:
        for li in fr.read().split("\n"):
            one_sol = li.strip().split(" ")[:-1]
            if len(li) <= 1:
                continue
            # print(li)
            sampled_assig = [1 if int(x) > 0 else 0 for x in one_sol]
            sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    print(len(sampled_assignments), sampled_assignments[0].shape)
    return sampled_assignments


def draw_from_unigen(num_samples, input_file):
    tmpfile = '/tmp/unigen.txt'
    cmd = """./src/lll/sampler/uniformSATSampler/unigen --input {} --samples {} --sampleout {} > /tmp/tmp.txt""".format(input_file,
                                                                                                                        num_samples,
                                                                                                                        tmpfile)

    os.system(cmd)
    sampled_assignments = []
    with open(tmpfile, 'r') as fr:
        for li in fr.read().split("\n"):
            one_sol = li.strip().split(" ")[:-1]
            # print(one_sol)
            if len(li) <= 1:
                continue
            sampled_assig = [1 if int(x) > 0 else 0 for x in one_sol]
            # print(sampled_assig)
            sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments
