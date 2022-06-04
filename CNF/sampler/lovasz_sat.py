from collections import defaultdict
import random
MAX = 1000000


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
    for it in range(MAX*number_clauses):
        # find the first violated clauses.
        violated_clause = None
        for clause in instance.clauses:
            if not is_satisfied(clause, assignment):
                violated_clause = clause
                break
        if violated_clause == None:
            return (assignment, it + 1)
        # Resample the variable
        for number in violated_clause:
            var = number >> 1
            assignment[var] = random.choice(values)
            
    
    return None, MAX*number_clauses


def partial_rejection_sampling_sampler(instance, values=[1, 0]):
    """
    Lovasz Parallel SAT Sampler.
    We start with a random assignment for all variables. Then, we find the all the violated clauses,
    and resample the variables in all those clause (assigning a new values to each of them IID).
    We repeat the process until we find a valid assignment.

    Mark Jerrum. Fundamentals of Partial Rejection Sampling. Arxiv 14 June 2021.
    URL: https://arxiv.org/pdf/2106.07744.pdf
    """
    number_clauses = len(instance.clauses)
    # start with a random assignment for all variables.
    assignment = [random.choice(values)
                  for _ in range(len(instance.variables))]
    for it in range(MAX*number_clauses):
        # find the first violated clauses.
        violated_clauses = []
        for clause in instance.clauses:
            if not is_satisfied(clause, assignment):
                violated_clauses.append(clause)

        if len(violated_clauses) == 0:
            return (assignment, it + 1)
        
        # Resample the variable
        all_vars = set()
        for clause in violated_clauses:
            for number in clause: 
                var = number >> 1
                all_vars.add(var)

        for var in list(all_vars):
            assignment[var] = random.choice(values)
    return None, MAX*number_clauses


def is_satisfied(clause, assignment):
    for number in clause:
        var = number >> 1
        neg = number & 1
        if assignment[var] != neg:
            return True
    return False
