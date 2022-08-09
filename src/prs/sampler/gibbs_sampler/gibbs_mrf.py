
import numpy as np
from prs.utils.load_uai import UaiFile
import copy
import warnings
warnings.filterwarnings("ignore")

def Gibbs_Sampling(filename, n_samples, initial=None, burnin_time=50):
    # Takes too much time
    uai = UaiFile(filename)
    if initial == None:
        term = np.random.binomial(1, 0.5, size=(n_samples, uai.n_var))
    else:
        term = copy.deepcopy(initial)

    # Vectorize
    for j in range(burnin_time):
        for x_idx in range(uai.n_var):
            # sample each element x
            p = np.ones([n_samples, 2], dtype=float)

            # compute marginal distribution
            for f_idx in range(uai.n_cliques):
                # each factor
                arr = np.array(uai.cliques[f_idx])
                loc = np.where(arr == x_idx)[0]
                if len(loc) == 0:
                    continue

                label = term[:, tuple(uai.cliques[f_idx])]
                label[:, loc[0]] = 0
                p[:, 0] *= uai.factors[f_idx][tuple(label.T)]
                label[:, loc[0]] = 1
                p[:, 1] *= uai.factors[f_idx][tuple(label.T)]

            p /= np.sum(p, axis=1)[:, None]

            rand_ = np.random.uniform(0.0, 1.0, size=(n_samples, 1))

            term[(rand_.squeeze() < p[:, 0].squeeze()), x_idx] = 0
            term[(rand_.squeeze() >= p[:, 0].squeeze()), x_idx] = 1


    return term
