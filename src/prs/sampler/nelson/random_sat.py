
import random
MAX = 1000000


def Monte_Carlo_sampler(instance, values=[True, False]):
    """
    Random Sat Solver (naive trial-and-error algorithm).
    We keep generating a random assignment until we obtain one valid assignment.
    """
    number_clauses = len(instance.clauses)

    # start with a random assignment for all variables.
    assignment = [random.choice(values)
                  for _ in range(len(instance.variables))]
    modified_var = None
    for it in range(MAX*number_clauses):
        # find the first violated clauses.
        violated_clause = None
        for clause in instance.clauses:
            if not is_satisfied(clause, assignment):
                violated_clause = clause
                break
        if violated_clause == None:
            print("Terminate after {} iterations".format(it))
            print(assignment, it + 1)
        else:
            assignment = [random.choice(values)
                          for _ in range(len(instance.variables))]
    return None, it+1


def is_satisfied(clause, assignment):
    for number in clause:
        var = number >> 1
        neg = number & 1
        if assignment[var] != neg:
            return True
        return False
