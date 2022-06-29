import random
import numpy as np

MAX = 5000
import torch
import time

def constructive_lovasz_local_lemma_sampler(instance, values=[1, 0]):
    """
    Lovasz Sequential SAT Sampler.
    We start with a random assignment for all variables. Then, we find the first violated clause,
    and resample the variables in that clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.

    Robin A. Moser, Gabor Tardos. A constructive proof of the general lovász local lemma. J. ACM 2010. 
    URL: https://dl.acm.org/doi/10.1145/1667053.1667060
    """
    number_clauses = len(instance.clauses)
    # start with a random assignment for all variables.
    assignment = [random.choice(values)
                  for _ in range(len(instance.variables))]
    for it in range(MAX * number_clauses):
        # find the first violated clauses.
        violated_clause = None
        for clause in instance.clauses:
            if not is_satisfied(clause, assignment):
                violated_clause = clause
                break
        if not violated_clause:
            return assignment, it + 1
        # Resample the variable
        for number in violated_clause:
            var = number >> 1
            assignment[var] = random.choice(values)

    return None, MAX * number_clauses


def conditioned_partial_rejection_sampling_sampler(instance, values=[1, 0]):
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
    number_clauses = len(instance.clauses)
    # start with a random assignment for all variables.
    assignment = [random.choice(values)
                  for _ in range(len(instance.variables))]
    for it in range(MAX * number_clauses):
        # find all the violated clauses.
        violated_clauses = []
        for clause in instance.clauses:
            if not is_satisfied(clause, assignment):
                violated_clauses.append(clause)

        if len(violated_clauses) == 0:
            return assignment, it + 1

        # Resample the variable
        all_vars = set()
        for clause in violated_clauses:
            for number in clause:
                var = number >> 1
                all_vars.add(var)

        for var in list(all_vars):
            assignment[var] = random.choice(values)
    return None, MAX * number_clauses




def is_satisfied(clause, assignment):
    for number in clause:
        var = number >> 1
        neg = number & 1
        if assignment[var] != neg:
            return True
    return False


def pytorch_conditioned_partial_rejection_sampling_sampler(instance, device, prob=0.5):

    """
    Implementation of Neural Lovasz Sampler using Pytorch, with extreme condition for the input SAT-CNF.
    We start with a random assignment for all variables. Then, we find the all the violated clauses,
    and resample the variables in all those clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.
    """
    clause2var, weight, bias = instance.get_formular_matrix_form()
    clause2var, weight, bias = torch.from_numpy(clause2var).float().to(device), torch.from_numpy(weight).float().to(device), torch.from_numpy(bias).float().to(device)

    st = time.time()
    # start with a random assignment for all variables.
    init_rand = torch.rand((len(instance.variables),)).to(device)
    assignment = (init_rand > prob).float().to(device)

    for it in range(MAX):

        # Compute all the clauses 
        Z = torch.einsum('ikj,j->ik', weight, assignment) + bias
        C, C_idxs = torch.max(Z, dim=1)
        violated_clause = (1 - C).reshape(1, -1)
        # Extracted all the random variable in the filtered clauses
        violated_RV = torch.matmul(violated_clause, clause2var).squeeze()
        if torch.sum(violated_RV) == 0:
            # print("done!")
            return assignment, it + 1, (time.time() - st)*1000
        resampled_rand = torch.rand((len(instance.variables),)).to(device)
        random_assignment = (resampled_rand > prob).float().to(device)

        assignment = torch.multiply((1 - violated_RV), assignment) + torch.multiply(violated_RV, random_assignment)
    # print("NaN time for sampler:", time.time() - st)
    return [], MAX, (time.time() - st)*1000



def numpy_conditioned_partial_rejection_sampling_sampler(instance, prob=0.5):
    """
    Lovasz SAT Sampler using Numpy implementation with extreme condition for the input SAT-CNF.
    We start with a random assignment for all variables. Then, we find the all the violated clauses,
    and resample the variables in all those clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.
    """
    number_clauses = len(instance.clauses)
    clause2var, weight, bias = instance.get_formular_matrix_form()

    # start with a random assignment for all variables.
    uniform_rand = np.random.rand(len(instance.variables))
    assignment = (uniform_rand > prob).astype(int)

    for it in range(MAX * number_clauses):
        # Compute all the clauses 
        Z = np.einsum('ikj,j->ik', weight, assignment) + bias
        C = np.max(Z, axis=1)
        violated_clause = (1 - C).reshape(1, -1)
        # Extracted all the random variable in the filtered clauses
        violated_RV = np.matmul(violated_clause, clause2var).squeeze()
        if np.sum(violated_RV) == 0:
            return assignment, it + 1
        uniform_rand = np.random.rand(len(instance.variables))
        random_assignment = (uniform_rand > prob).astype(int)
        assignment = np.multiply((1 - violated_RV), assignment) + np.multiply(violated_RV, random_assignment)

    return [], MAX
