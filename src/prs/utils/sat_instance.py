import numpy
import numpy as np
from pysat.formula import CNF

MAX = 5000

import random
from pysat.solvers import Solver


# K-SAT problem. K is the number of literal in every clause


class SAT_Instance(object):
    def __init__(self, K, cnf_file=None):
        self.variables = []
        self.variable_table = dict()

        self.clauses = []
        self.clause2var = None  # matrix for mapping clause 2 variables
        self.weight = None
        self.bias = None
        self.K = K
        self.cnf = CNF(cnf_file)

    def parse_and_add_clause(self, line):
        clause = []
        literals = line.split()
        if literals[-1].strip() == '0':
            literals = literals[:-1]
        for literal in literals:

            if literal.startswith("-"):
                negated = 1
            else:
                negated = 0
            variable = literal[negated:]
            if variable not in self.variable_table:
                self.variable_table[variable] = len(self.variable_table) + 1

            if negated:
                encoded_literal = - self.variable_table[variable]
            else:
                encoded_literal = self.variable_table[variable]
            clause.append(encoded_literal)
        new_clause = tuple(set(clause))
        self.clauses.append(new_clause)
        self.cnf.append(new_clause)

    @classmethod
    def from_cnf_file(cls, input_file, K=-1):
        instance = None
        if K > 1:
            instance = cls(K)
        with open(input_file, 'r') as file:

            for line in file:
                line = line.strip()
                if len(line) < 1:
                    continue
                if 'clause length' in line:
                    K = int(line.split("=")[-1].strip())
                    print("init {}-sat instance".format(K))
                    instance = cls(K)
                    continue
                if not line.startswith("c") and not line.startswith("p") and len(line) > 1:
                    instance.parse_and_add_clause(line)
        instance.variables = list(instance.variable_table.keys())
        # print("clauses: {}, RVs {}".format(len(instance.cnf.clauses), instance.cnf.nv))

        return instance

    def get_formular_matrix_form(self):
        self.weight = np.zeros(
            (len(self.cnf.clauses), self.K, self.cnf.nv), dtype=int)
        self.bias = np.zeros((len(self.clauses), self.K), dtype=int)

        for i in range(len(self.clauses)):
            ci = self.clauses[i]

            for j in range(self.K):
                random_variable = ci[j]
                if random_variable > 0:
                    self.weight[i, j, random_variable - 1] = 1
                    self.bias[i, j] = 0
                else:
                    self.weight[i, j, -random_variable - 1] = -1
                    self.bias[i, j] = 1

        self.clause2var = np.squeeze(np.sum(np.abs(self.weight), axis=1))
        # print("M={}, W={}, b={}".format(self.clause2var.shape, self.weight.shape, self.bias.shape))

        return self.clause2var, self.weight, self.bias

def get_all_solutions(instance):
    all_solutions = []
    with Solver(name='glucose4', bootstrap_with=instance.cnf.clauses, use_timer=True) as s:
        for one_solution in s.enum_models():
            np_one_solution = np.asarray(one_solution) > 0
            np_one_solution = np_one_solution.astype(int)
            all_solutions.append(np_one_solution)
    return all_solutions

def generate_random_solutions_with_preference(instance, number_of_random_valid_solutions=200):
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

    solutions = enumerate_models(instance.cnf, number_of_random_valid_solutions, 'glucose4')
    preferred = solutions[:(len(solutions) // 2)]
    less_preferred = solutions[(len(solutions) // 2):]
    return preferred, less_preferred



