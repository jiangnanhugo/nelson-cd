import numpy as np
from pysat.formula import CNF

MAX = 5000


# K-SAT problem. K is the number of literal in every clause

def cnf_to_uai(cnf_formula: CNF, prob: np.array, filename):
    with open(filename, 'w') as fw:
        fw.write("MARKOV\n")
        fw.write("{}\n".format(cnf_formula.nv))
        fw.write(" ".join(['2' for _ in range(cnf_formula.nv)]) + '\n')

        # layer X={x_i}_{i=1}^n
        fw.write("{}\n".format(cnf_formula.nv + len(cnf_formula.clauses)))
        for i in range(cnf_formula.nv):
            fw.write("{} {}\n".format(1, i))
        # layer C={c_j}_{j=1}^m
        for ci in cnf_formula.clauses:
            str_ci = [str(abs(xj) - 1) for xj in ci]
            fw.write("{} {}\n".format(len(str_ci), " ".join(str_ci)))

        # prob of X={x_i}_{i=1}^n
        for i in range(cnf_formula.nv):
            fw.write("2\n")
            fw.write("{}\n{}\n".format(prob[i], 1 - prob[i]))

        # Truth table of C={c_j}_{j=1}^m
        for ci in cnf_formula.clauses:
            fw.write("{}\n".format(2 ** len(ci)))
            for assignment in range(2 ** len(ci)):
                binarized = "{0:b}".format(assignment).zfill(len(ci))
                output = 0.000
                for j, xj in enumerate(ci):
                    if (xj < 0 and binarized[j] == '0') or (xj > 0 and binarized[j] == '1'):
                        output = 1.0000
                        break
                fw.write("{:.4f}\n".format(output))


if __name__ == '__main__':
    cnf = CNF("/home/jiangnan/PycharmProjects/partial-rejection-sampling/datasets/rand-k-sat/3_5_3/randkcnf_3_5_3_0001.cnf")
    prob = np.random.rand(cnf.nv)
    cnf_to_uai(cnf, prob, "output.uai")
