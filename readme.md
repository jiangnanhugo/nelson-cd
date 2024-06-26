# Learning Markov Random Fields for Combinatorial Structures via Sampling through Lov\'asz Local Lemma #

full paper: https://arxiv.org/abs/2212.00296


## Requirements
the code needs to install pytroch
```bash
pip install torch
```

### Datasets
to generate the Rand-k-cnf dataset we need
```python
pip install cnfgen
pip install 'python-sat[pblib,aiger]'
```

## 1. Uniform sampling comparison

sampling SAT-CNF solutions uniformly.

```bash
./run_weighted_random_ksat.py lll 5 300 300 uniform
./run_weighted_random_ksat.py lll 5 300 300 weighted
```


```bash
[nelson, spur, unigen, quciksampler, cmsgen, bddsampler, kus, smarch, searchtreesampler]
```

note that, for `unigen, cmsgen` you need to goto its repository and install the program according to the `readme.md` file.

To run the `kus` program, you need to install `pydot`.

All the baseline methods included for comparison are cloned from its public-available repositories. If you want to run the rest programs, you need to goto every link and install it.


- [Spur](https://github.com/ZaydH/spur)
- [Unigen](https://github.com/meelgroup/unigen)
- [QuickSampler](https://github.com/RafaelTupynamba/quicksampler)
- [CMSGen](https://github.com/meelgroup/cmsgen)
- [KUS](https://github.com/meelgroup/KUS). 
- [Smarch](https://github.com/jeho-oh/Smarch)
- [STS](https://cs.stanford.edu/~ermon/code/STS.zip)

### 2. Weighted sampling comparison

 **Requirement**:
 ```python
pip install waps
```
