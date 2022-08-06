import time
import torch
import math
import os
import numpy as np
from waps import sampler as waps_sampler

from prs.sampler.nelson.lovasz_sat import pytorch_neural_lovasz_sampler, constructive_lovasz_local_lemma_sampler, \
    partial_rejection_sampling_sampler, pytorch_batch_neural_lovasz_sampler, numpy_neural_lovasz_sampler
from prs.sampler.nelson.random_sat import Monte_Carlo_sampler
from prs.sampler.xor_sampling.xor_sampler import XOR_Sampling
from prs.utils.cnf2uai import cnf_to_uai


def xor_sample(cnf_instance, prob, input_file, num_samples):
    cnf_to_uai(cnf_instance, prob, input_file + ".weight.uai")
    returned_samples = XOR_Sampling(input_file + ".weight.uai", num_samples)


def draw_from_weightgen(num_samples, input_file, instance, prob, device='cuda'):
    instance.cnf.to_file(input_file + ".weight")
    with open(input_file + ".weight", "a") as fw:
        for xi in range(instance.cnf.nv):
            fw.write("w {} {} 0\n".format(xi + 1, prob[xi]))
            fw.write("w -{} {} 0\n".format(xi + 1, 1.0 - prob[xi]))
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
    cmd = """./src/prs/sampler/weightedSATSampler/weightgen --samples={} --kappa={} --pivotUniGen={} --maxTotalTime={} \
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


def draw_from_prs_series(algo, Fi, clause2var, weight, bias, prob, num_samples, device='cuda'):
    if algo == 'lll':
        sampler = constructive_lovasz_local_lemma_sampler
    elif algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif algo == 'prs':
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
        # prob = torch.ones(Fi.cnf.nv).to(device) * 0.5
        clause2var = torch.transpose(clause2var, 0, 1)
        clause2var = clause2var.reshape(1, *clause2var.size())
        weight = weight.reshape(1, *weight.size())
        bias = bias.reshape(1, *bias.size())
        assignments, count, _ = pytorch_batch_neural_lovasz_sampler(Fi, clause2var, weight, bias, device=device, prob=prob,
                                                                    batch_size=num_samples)
        if len(assignments) > 0:
            samples = assignments

    return samples


def draw_from_waps(num_samples, input_file, instance, prob, device='cuda'):
    instance.cnf.to_file(input_file + ".weight")
    with open(input_file + ".weight", "a") as fw:
        for xi in range(instance.cnf.nv):
            fw.write("w {} {} 0\n".format(xi + 1, prob[xi]))
            fw.write("w -{} {} 0\n".format(xi + 1, 1.0 - prob[xi]))
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


def draw_from_quicksampler(num_samples, input_file, device='cuda'):
    cmd = """./src/prs/sampler/uniformSATSampler/quicksampler -n {} -t 180.0 {} >/tmp/tmp.log""".format(num_samples, input_file)

    os.system(cmd)
    sampled_assignments = []
    with open(input_file + '.samples', 'r') as fr:
        idx = 0
        for li in fr.read().split("\n"):
            one_sol = li.strip().split(" ")[-1].strip()
            if len(one_sol) <= 1:
                continue
            if idx >= num_samples:
                break
            sampled_assig = [int(x) for x in one_sol]
            sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
            idx += 1
    return sampled_assignments


def draw_from_kus(num_samples, input_file, device='cuda'):
    tmpfile = "/tmp/randksat.kus.txt"
    cmd = "python3  ./src/prs/sampler/uniformSATSampler/KUS.py --samples {} --outputfile {} {} >/tmp/tmp.log".format(num_samples,
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


def draw_from_cmsgen(num_samples, input_file, device='cuda'):
    tmpfile = "/tmp/randksat.cmsgen.txt"
    cmd = "./src/prs/sampler/uniformSATSampler/cmsgen --samples {} --samplefile {} {} >/tmp/tmp.log".format(num_samples,
                                                                                                            tmpfile,
                                                                                                            input_file)
    os.system(cmd)
    sampled_assignments = []
    with open(tmpfile, 'r') as fr:
        for li in fr.read().split("\n"):
            one_sol = li.strip().split(" ")[:-1]
            # print(one_sol)
            if len(li) <= 1:
                continue
            sampled_assig = [1 if int(x) > 0 else -1 for x in one_sol]
            sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments


def draw_from_unigen(num_samples, input_file, device='cuda'):
    tmpfile = '/tmp/unigen.txt'
    cmd = """./src/prs/sampler/uniformSATSampler/unigen --input {} --samples {} --sampleout {} > /tmp/tmp.txt""".format(input_file,
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
            sampled_assig = [1 if int(x) > 0 else -1 for x in one_sol]
            # print(sampled_assig)
            sampled_assignments.append(torch.from_numpy(np.array(sampled_assig)).to(device).reshape(1, -1))
    return sampled_assignments
