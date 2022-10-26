# Hierarchical classification at multiple operating points

https://arxiv.org/abs/2210.10929

This repo contains the code for the NeurIPS 2022 paper.

For now, it's in a raw state.
I will refactor the code and remove legacy parts when I have time!

The main functionality can be found in:

* [hier.py](hier.py): Representation of class hierarchy and basic functionality.
* [hier_torch.py](hier_torch.py): Prediction and loss functions written in pytorch.
* [infer.py](infer.py): Inference functions that map likelihoods to labels.
* [metrics.py](metrics.py): Functions to evaluate metrics.

Code to run experiments and generate plots:

* [main.py](main.py)
* [eval_inat21mini.ipynb](eval_inat21mini.ipynb)
* [eval_inat21_errorbars.ipynb](eval_inat21_errorbars.ipynb)
* [eval_inat21_split.ipynb](eval_inat21_split.ipynb)
* [eval_imagenet.ipynb](eval_imagenet.ipynb)
