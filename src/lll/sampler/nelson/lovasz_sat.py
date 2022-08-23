import random
import numpy as np
from opt_einsum import contract

MAX = 5000
import torch
import time


def constructive_lovasz_local_lemma_sampler(cnf_instance, prob=None, max_tryouts=10000, **kwargs):
    """
    Lovasz Sequential SAT Sampler.
    We start with a random assignment for all variables. Then, we find the first violated clause,
    and resample the variables in that clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.

    Robin A. Moser, Gabor Tardos. A constructive proof of the general lovász local lemma. J. ACM 2010. 
    URL: https://dl.acm.org/doi/10.1145/1667053.1667060
    """
    number_clauses = len(cnf_instance.clauses)
    # start with a random assignment for all variables.

    assignment = [random.random() > prob[i]
                  for i in range(len(cnf_instance.variables))]
    st = time.time()
    for it in range(max_tryouts * number_clauses):
        # find the first violated clauses.
        violated_clause = None
        for clause in cnf_instance.clauses:
            if not is_satisfied(clause, assignment):
                violated_clause = clause
                break
        if not violated_clause:
            return assignment, it + 1, time.time() - st
        # Resample the variable
        for number in violated_clause:
            var = number >> 1
            assignment[var] = random.random() > prob[var]

    return [], MAX * number_clauses, time.time() - st


def partial_rejection_sampling_sampler(cnf_instance, prob=None, max_tryouts=10000, **kwargs):
    """
    Lovasz Parallel SAT Sampler with extreme condition for the input SAT-CNF..
    We start with a random assignment for all variables. Then, we find the all the violated clauses,
    and resample the variables in all those clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.

    Heng Guo, Mark Jerrum and Jingcheng Liu. Uniform sampling through the Lovász local lemma. J. ACM, 2019. 
    URL: https://dl.acm.org/doi/10.1145/3310131
    Mark Jerrum. Fundamentals of Partial Rejection Sampling. Arxiv 14 June 2021.
    URL: https://arxiv.org/pdf/2106.07744.pdf
    """
    number_clauses = len(cnf_instance.clauses)
    # start with a random assignment for all variables.
    assignment = [random.random() > prob[i]
                  for i in range(len(cnf_instance.variables))]
    st = time.time()
    for it in range(max_tryouts * number_clauses):
        # find all the violated clauses.
        violated_clauses = []
        for clause in cnf_instance.clauses:
            if not is_satisfied(clause, assignment):
                violated_clauses.append(clause)

        if len(violated_clauses) == 0:
            return assignment, it + 1, time.time() - st

        # Resample the variable
        all_vars = set()
        for clause in violated_clauses:
            for number in clause:
                var = number >> 1
                all_vars.add(var)

        for var in list(all_vars):
            assignment[var] = random.random() > prob[var]
    return [], MAX * number_clauses, time.time() - st


def is_satisfied(clause, assignment):
    for number in clause:
        var = number >> 1
        neg = number & 1
        if assignment[var] != neg:
            return True
    return False


def pytorch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias, device, prob=0.5, max_tryouts=10000, **kwargs):
    """
    Implementation of Neural Lovasz Sampler using Pytorch, with extreme condition for the input SAT-CNF.
    We start with a random assignment for all variables. Then, we find the all the violated clauses,
    and resample the variables in all those clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.
    """

    st = time.time()
    # start with a random assignment for all variables.
    init_rand = torch.rand((cnf_instance.nv,)).to(device)
    assignment = (init_rand > prob).int().to(device)

    for it in range(max_tryouts):
        # Compute all the clauses 
        Z = torch.sum(torch.mul(weight, assignment), dim=-1) + bias
        C, C_idxs = torch.max(Z, dim=1)
        violated_clause = (1 - C).reshape(1, -1)
        if torch.sum(violated_clause) == 0:
            return assignment, it + 1, time.time() - st
        # Extracted all the random variable in the filtered clauses
        violated_RV = torch.sum(torch.mul(violated_clause.reshape(-1), torch.transpose(clause2var, 0, 1)), dim=1)
        violated_RV = (violated_RV > 0).int()

        resampled_rand = torch.rand((cnf_instance.nv,)).to(device)
        random_assignment = (resampled_rand > prob).int().to(device)

        assignment = torch.mul((1 - violated_RV), assignment) + torch.mul(violated_RV, random_assignment)
    # print("NaN time for sampler:", time.time() - st)
    return [], MAX, time.time() - st


def pytorch_batch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias, device, prob=0.5, max_tryouts=10000, **kwargs):
    """
    A batch version of the above function
    """
    batch_size = kwargs['batch_size']
    st = time.time()
    # start with a random assignment for all variables.
    init_rand = torch.rand((batch_size, cnf_instance.nv)).to(device)
    assignment = (init_rand > prob).int()
    for it in range(max_tryouts):
        O = torch.mul(weight, assignment.reshape(assignment.size()[0], 1, 1, assignment.size()[1]))
        Z = torch.sum(O, dim=-1) + bias

        C, _ = torch.max(Z, dim=-1)
        violated_clause = 1 - C
        if torch.sum(violated_clause) == 0:
            return assignment, it + 1, time.time() - st
        # Extracted all the random variable in the filtered clauses
        violated_RV = torch.sum(torch.mul(violated_clause.reshape(violated_clause.shape[0], 1, violated_clause.shape[1]), clause2var), dim=-1)
        violated_RV = (violated_RV > 0).int()
        resampled_rand = torch.rand((batch_size, cnf_instance.nv)).to(device)

        random_assignment = (resampled_rand > prob).int()

        assignment = torch.mul((1 - violated_RV), assignment) + torch.mul(violated_RV, random_assignment)

    return [], MAX, time.time() - st


def numpy_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias, prob=0.5, max_tryouts=10000, **kwargs):
    """
    Lovasz SAT Sampler using Numpy implementation with extreme condition for the input SAT-CNF.
    We start with a random assignment for all variables. Then, we find the all the violated clauses,
    and resample the variables in all those clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.
    """
    # clause2var, weight, bias = instance.get_formular_matrix_form()
    # start with a random assignment for all variables.
    uniform_rand = np.random.rand(cnf_instance.nv)
    if uniform_rand.shape[0] != prob.shape[0]:
        print(uniform_rand.shape, prob.shape)
    assignment = (uniform_rand > prob).astype(int)

    st = time.time()
    for it in range(max_tryouts):
        # Compute all the clauses 
        Z = np.einsum('ikj,j->ik', weight, assignment) + bias
        C = np.max(Z, axis=1)
        violated_clause = (1 - C).reshape(1, -1)
        # Extracted all the random variable in the filtered clauses
        violated_RV = np.matmul(violated_clause, clause2var).squeeze()
        violated_RV = violated_RV > 0
        if np.sum(violated_RV) == 0:
            return assignment, it + 1, time.time() - st
        uniform_rand = np.random.rand(cnf_instance.nv)
        if uniform_rand.shape[0] != prob.shape[0]:
            print(uniform_rand.shape, prob.shape)
        random_assignment = (uniform_rand > prob).astype(int)
        assignment = np.multiply((1 - violated_RV), assignment) + np.multiply(violated_RV, random_assignment)

    return [], MAX, time.time() - st


def numpy_batch_neural_lovasz_sampler(cnf_instance, clause2var, weight, bias, prob=0.5, max_tryouts=10000, **kwargs):
    """
    Lovasz SAT Sampler using Numpy implementation with extreme condition for the input SAT-CNF.
    We start with a random assignment for all variables. Then, we find the all the violated clauses,
    and resample the variables in all those clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.
    """
    batch_size = kwargs['batch_size']
    # start with a random assignment for all variables.
    init_rand = np.random.rand(batch_size, cnf_instance.nv)
    # if init_rand.shape[-1] != prob.shape[0]:
    #     print(init_rand.shape, prob.shape)
    assignment = init_rand > prob

    st = time.time()
    for it in range(max_tryouts):
        # Compute all the clauses
        Z = contract('ikj,lj->lik', weight, assignment) + bias
        violated_clause = 1 - np.max(Z, axis=-1)
        # Extracted all the random variable in the filtered clauses
        violated_RV = np.matmul(violated_clause, clause2var).squeeze()
        violated_RV = violated_RV > 0
        if np.sum(violated_RV) == 0:
            return assignment, it + 1, time.time() - st
        uniform_rand = np.random.rand(batch_size, cnf_instance.nv)
        # if uniform_rand.shape[-1] != prob.shape[0]:
        #     print(uniform_rand.shape, prob.shape)
        random_assignment = uniform_rand > prob
        assignment = np.multiply(1 - violated_RV, assignment) + np.multiply(violated_RV, random_assignment)

    return [], MAX, time.time() - st
