# Invariant-Representation-for-Continual-Learning
This is the official PyTorch implementation for the [Learning Invariant Representation for Continual Learning](https://arxiv.org/abs/2101.06162) paper at the AAAI Meta-Learning for Computer Vision Workshop (2021).

# Abstract
We propose a new pseudo-rehearsal-based method, named learning Invariant Representation for Continual Learning (IRCL), in which class-invariant representation is disentangled from a conditional generative model and jointly used with class-specific representation to learn the sequential tasks. Disentangling the shared invariant representation helps to learn continually a sequence of tasks, while being more robust to forgetting and having better knowledge transfer. We focus on class incremental learning where there is no knowledge about task identity during inference.

# Requirements
* Python 3.6
* Pytorch 1.2
* torchvision 0.4

# Usage
You can use main.py to run our IRCL method on the Split MNIST benchmark. 

```
python main.py
```

# Reference
If you use this code, please cite our paper:
