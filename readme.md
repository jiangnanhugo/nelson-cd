### Experiments of Neural Lovasz Sampler

### 1. Uniform sampling comparison

sampling SAT-CNF solutions uniformly.

```bash
PATH_TO_ONE_CNF_FILE=dataset/random-k-sat/5-30-30/*.cnf
cd uniform_sample_evaluate
python run_evaluation.py --algo nelson --input_cnf PATH_TO_ONE_CNF_FILE
```

notice that `--algo` means choose the sampler. Here we can choose among the following list of algorithms:

```bash
[nelson, spur, unigen3, quciksampler, cmsgen, bddsampler, kus, smarch, searchtreesampler]
```

All the baseline methods included for comparison are cloned from its public-available repositories. 
- [Spur](https://github.com/ZaydH/spur)
- [Unigen3](https://github.com/meelgroup/unigen)
- [QuickSampler](https://github.com/RafaelTupynamba/quicksampler)
- [CMSGen](https://github.com/meelgroup/cmsgen)
- [KUS](https://github.com/meelgroup/KUS). 
- [Smarch](https://github.com/jeho-oh/Smarch)
- [STS](https://cs.stanford.edu/~ermon/code/STS.zip)

### 2. Weighted sampling comparison
