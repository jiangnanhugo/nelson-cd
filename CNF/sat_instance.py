"""
notations on encoding:
- Variables: encoded as numbers 0 to n-1;
- Literal l: v is 2*v and ~v is 2*v+1. so the foremost bit of a literal encodes whether it is negated or not.
           This can be tested simply checking if l&1 is 0 or 1.
- To negate a literal, we jsut have to toggle the foremost bit. THis can be done by XOR with 1:
  The negation of l is: l XOR 1.
- To get a lieral's variable, we just need to shift to the right. THis can be done with l >>1.

E.g., variable b is encoded with number 3. The literal b is encoded as: 2*3 = 6 and ~b as 2*3+1=7
"""
import numpy as np
from pysat.formula import CNF

import random

MAX = 5000

import random
from pysat.solvers import Lingeling, Glucose4
import getopt
import os
from pysat.formula import CNFPlus
from pysat.solvers import Solver, SolverNames
import sys


# from pysat.solvers import Glucose3
from pysat.solvers import Lingeling

# K-SAT problem. K is the number of literal in every clause


class SAT_Instance(object):
    def __init__(self, K, cnf_file=None):
        self.variables = []
        self.variable_table = dict()

        self.clauses = []
        self.clause2var = None  # matrix for maping cluase 2 variables
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
    def from_file(cls, input_file, K=3):
        with open(input_file, 'r') as file:
            instance = cls(K)
            for line in file:
                line = line.strip()
                if len(line) > 0 and not line.startswith("#"):
                    instance.parse_and_add_clause(line)

        instance.variables = list(instance.variable_table.keys())
        print("clauses: {}".format(instance.cnf.clauses))
        print("# of variables:", instance.cnf.nv)
        return instance


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
            (len(self.clauses), self.K, len(self.variables)), dtype=int)
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


def enumerate_models(formula, to_enum, solver):
    """
        Enumeration procedure. It represents a loop iterating over satisfying
        assignment for a given formula until either all or a given number of
        them is enumerated.
        :param formula: input WCNF formula
        :param to_enum: number of models to compute
        :param solver: name of SAT solver
        :type formula: :class:`.CNFPlus`
        :type to_enum: int or 'all'
        :type solver: str
    """
    all_solutions = []
    with Solver(name=solver, bootstrap_with=formula.clauses, use_timer=True) as s:
        for i, one_solution in enumerate(s.enum_models(), 1):
            np_one_solution = np.asarray(one_solution) > 0
            np_one_solution = np_one_solution.astype(int)
            all_solutions.append(np_one_solution)
            if i == to_enum:
                random.shuffle(all_solutions)
                return all_solutions


def generate_random_solutions_with_preference(instances, number_of_random_valid_solutions=20):
    solutions_and_formulas = []
    for inst in instances:
        solutions = enumerate_models(inst.cnf, number_of_random_valid_solutions, 'glucose4')
        preferred = solutions[:(len(solutions) // 2)]
        less_preferred = solutions[(len(solutions) // 2):]
        solutions_and_formulas.append((preferred, less_preferred, inst))

