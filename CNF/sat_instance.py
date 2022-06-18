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


# from pysat.solvers import Glucose3
from pysat.solvers import Lingeling

# K-SAT problem. K is the number of literal in every clause


class SAT_Instance(object):
    def __init__(self, K):
        self.variables = []
        self.variable_table = dict()
        # self.variable_table[0]=0
        
        self.clauses = []
        self.clause2var = None   # matrix for maping cluase 2 variables
        self.weight = None
        self.bias = None
        self.K = K
        self.cnf = CNF()

    def parse_and_add_clause(self, line):
        clause = []
        for literal in line.split():
            if literal.startswith("-"):
                negated = 1
            else:
                negated = 0
            variable = literal[negated:]
            if variable not in self.variable_table:
                self.variable_table[variable] = len(self.variable_table)+1
                
            if negated:
                encoded_literal = - self.variable_table[variable]
            else:
                encoded_literal = self.variable_table[variable]
            clause.append(encoded_literal)
        new_clause = tuple(set(clause))
        self.clauses.append(new_clause)
        self.cnf.append(new_clause)

    @classmethod
    def from_file(cls, file, K=3):
        instance = cls(K)
        for line in file:
            line = line.strip()
            if len(line) > 0 and not line.startswith("#"):
                instance.parse_and_add_clause(line)
        instance.variables = list(instance.variable_table.keys())
        print("clauses: {}".format(instance.cnf.clauses))
        print("# of variables:", instance.cnf.nv)
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
                    self.weight[i, j, random_variable-1] = 1
                    self.bias[i, j] = 0
                else:
                    self.weight[i, j, -random_variable-1] = -1
                    self.bias[i, j] = 1

        self.clause2var = np.squeeze(np.sum(np.abs(self.weight), axis=1))
        print("M={}, W={}, b={}".format(
            self.clause2var.shape, self.weight.shape, self.bias.shape))
        return self.clause2var, self.weight, self.bias


def generate_random_solutions_with_preference(instances):
    solutions_and_formulas = []
    for inst in instances:
        with Lingeling(bootstrap_with=inst.cnf.clauses) as l:
            if l.solve() == True:
                one_solution=l.get_model()
                np_one_solution=np.asarray(one_solution)>0
                np_one_solution = np_one_solution.astype(int)
                print(np_one_solution)
                solutions_and_formulas.append([np_one_solution,inst])
    return solutions_and_formulas
