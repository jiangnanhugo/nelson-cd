# Experiments of Neural Lovasz Sampler


## Requirements
The script relies on zsh, you need to install it first.
```bash
sudo apt install zsh
chsh -s $(which zsh)
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

install pytroch
```bash
pip install torch
```

to generate the Rand-k-cnf dataset we need
```python
pip install cnfgen
pip install 'python-sat[pblib,aiger]'
```

## 1. Uniform sampling comparison

sampling SAT-CNF solutions uniformly.

```bash
./run_weighted_random_ksat.py prs 5 300 300 uniform
./run_weighted_random_ksat.py prs 5 300 300 weighted
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
