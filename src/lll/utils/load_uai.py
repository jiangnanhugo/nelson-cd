import numpy as np
import re


class UaiFile(object):
    def __init__(self, filename):
        self.filename = filename

        self.inst_type = ""
        self.n_var = 0
        self.dims = []
        self.n_cliques = 0
        self.cliques = []
        self.factors = []
        self.readUai(filename)
        return

    def readFileByTokens(self, path, specials=[]):
        spliton = '([\s' + ''.join(specials) + '])'
        with open(path, 'r') as fp:
            for line in fp:
                tok = [t.strip() for t in re.split(spliton, line) if t and not t.isspace()]
                for t in tok: yield t

    def readUai(self, filename):
        dims = []  # store dimension (# of states) of the variables
        cliques = []  # cliques (scopes) of the factors we read in
        factors = []  # the factors themselves

        gen = self.readFileByTokens(filename, '(),')  # get token generator for the UAI file
        inst_type = next(gen)
        n_var = int(next(gen))  # get the number of variables
        dims = [int(next(gen)) for i in range(n_var)]  # and their dimensions (states)
        n_cliques = int(next(gen))  # get the number of cliques / factors
        cliques = [None] * n_cliques
        for c in range(n_cliques):
            c_size = int(next(gen))  # (size of clique)
            cliques[c] = [int(next(gen)) for i in range(c_size)]

        factors = [None] * n_cliques
        for c in range(n_cliques):  # now read in the factor tables:
            t_size = int(next(gen))  # (# of entries in table = # of states in scope)
            factor_size = tuple(dims[v] for v in cliques[c]) if len(cliques[c]) else (1,)
            f_table = np.empty(t_size)
            for i in range(t_size):
                f_table[i] = float(next(gen))
            f_table = f_table.reshape(factor_size)

            f_table_T = np.transpose(f_table, tuple(np.argsort(cliques[c])))
            # factors[c] = tuple((cliques[c], np.array(f_table_T, dtype=float)))
            factors[c] = np.array(f_table_T, dtype=float)

        self.inst_type = inst_type
        self.n_var = n_var
        self.dims = dims
        self.n_cliques = n_cliques
        self.cliques = cliques
        self.factors = factors

        return factors, n_var

    def writeUai(self, filename):
        with open(filename, 'w') as fp:
            fp.write(self.inst_type + "\n")
            fp.write("{:d}\n".format(self.n_var))  # number of variables in model
            fp.write(" ".join(map(str, self.dims)) + "\n")  # write dimensions of each variable
            fp.write("{:d}\n".format(self.n_cliques));  # number of factors
            for clique in self.cliques:
                fp.write(str(len(clique)) + " " + " ".join(map(str, clique)))
                fp.write("\n")
            fp.write("\n")
            for factor in self.factors:
                fp.write(str(factor.size) + "\n")
                fp.write(str(factor).replace(' [', '').replace('[', '').replace(']', ''))
                fp.write("\n\n")


# if __name__ == '__main__':
#     uai = UaiFile("write_test.uai")
#     # testInstances/gridmod_attractive_n8_w3.0_f0.45.uai
#     # test.uai
#     # gridmod_attractive_n8_w3.0_f0.45.uai
#     # grid3x3.uai
#
#     print(uai.factors)
#     print("len")
#     print(len(uai.factors))
#     uai.writeUai("write_test2.uai")
