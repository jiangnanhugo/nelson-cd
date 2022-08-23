import numpy as np
import random

from pysat.formula import CNF
from pysat.solvers import Solver

MAX = 5000


# K-SAT problem. K is the number of literal in every clause


def get_formula_matrix_form(cnf_formula: CNF, K: int):
    W = np.zeros((len(cnf_formula.clauses), K, cnf_formula.nv), dtype=int)
    b = np.zeros((len(cnf_formula.clauses), K), dtype=int)

    for i in range(len(cnf_formula.clauses)):
        ci = cnf_formula.clauses[i]

        for j in range(K):
            random_variable = ci[j]
            if random_variable > 0:
                W[i, j, random_variable - 1] = 1
                b[i, j] = 0
            else:
                W[i, j, -random_variable - 1] = -1
                b[i, j] = 1

    M = np.squeeze(np.sum(np.abs(W), axis=1))
    return M, W, b


def get_all_solutions(cnf_formlula):
    all_solutions = []
    with Solver(name='glucose4', bootstrap_with=cnf_formlula.clauses, use_timer=True) as s:
        for one_solution in s.enum_models():
            np_one_solution = np.asarray(one_solution) > 0
            np_one_solution = np_one_solution.astype(int)
            all_solutions.append(np_one_solution)
    return all_solutions


def generate_random_solutions_with_preference(cnf_formlula, number_of_random_valid_solutions=200):
    def enumerate_models(formula, to_enum, solver):
        all_solutions = []
        with Solver(name=solver, bootstrap_with=formula.clauses, use_timer=True) as s:
            for i, one_solution in enumerate(s.enum_models(), 1):
                np_one_solution = np.asarray(one_solution) > 0
                np_one_solution = np_one_solution.astype(int)
                all_solutions.append(np_one_solution)
                if i == to_enum:
                    random.shuffle(all_solutions)
                    return all_solutions
            return all_solutions

    solutions = enumerate_models(cnf_formlula, number_of_random_valid_solutions, 'glucose4')
    preferred = solutions[:(len(solutions) // 2)]
    less_preferred = solutions[(len(solutions) // 2):]
    return preferred, less_preferred
