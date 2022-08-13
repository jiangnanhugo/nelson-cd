import os
import numpy as np
from lll.utils.load_uai import UaiFile


def XOR_Sampling(filename, n_samples, log_folder=''):
    uai = UaiFile(filename)
    term = []
    nbauxv = 10

    running_log = os.path.join(log_folder, "/tmp/tmp_paws_running_log.txt")
    res_log = os.path.join(log_folder, "/tmp/tmp_paws_res_log.txt")

    while len(term) < n_samples:
        os.system('./src/lll/sampler/xor_sampling/PAWS/paws -paritylevel 1 -timelimit 180 -samples {} -nbauxv {} -alpha 1 -pivot 8 -b 1  -outpath {}  {}> {}'
                  .format(n_samples, nbauxv, res_log, filename,  running_log))

        with open(res_log, 'r') as f:
            _ = f.readline()
            while True:
                line = f.readline()
                if len(line) == 0 or len(term) == n_samples:
                    break
                line = line[:uai.n_var]
                print(list(line))
                term.append(list(line))
    res = np.asarray(term, dtype=int)
    return res