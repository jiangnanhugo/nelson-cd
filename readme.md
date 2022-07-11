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



### 2. Weighted sampling comparison
