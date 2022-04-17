# NeuroNLP2
Deep neural models for core NLP tasks based on Pytorch(version 2)

This is the code we used in the following papers

>[Neural Probabilistic Model for Non-projective MST Parsing](http://www.cs.cmu.edu/~xuezhem/publications/IJCNLP2017.pdf)

>Xuezhe Ma, Eduard Hovy

>IJCNLP 2017

It also includes the re-implementation of the Stanford Deep BiAffine Parser:
>[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)

>Timothy Dozat, Christopher D. Manning

>ICLR 2017

## Updates
1. Upgraded the code to support PyTorch 1.3 and Python 3.6
2. Re-factored code to better organization

## Requirements

Python 3.6, PyTorch >=1.3.1, Gensim >= 0.12.0

## Data format
For the data format used in our implementation, please read this [issue](https://github.com/XuezheMax/NeuroNLP2/issues/9).

## Running the experiments
First to the experiments folder:

    cd experiments

### Dependency Parsing
To train a Deep BiAffine parser, simply run

    ./scripts/run_deepbiaf.sh
Again, remember to setup the paths for data and embeddings.

To train a Neural MST parser, 

    ./scripts/run_neuromst.sh
