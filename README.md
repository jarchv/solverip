Generative Flows as a General Purpose Solution for Inverse Problems
===================================================================

This repository contains the official Pytorch implementation of the paper:

>*Generative Flows as a General Purpose Solution for Inverse Problems* <br>
>José A. Chávez <br>
>https://arxiv.org/abs/2110.13285) <br>

>**Abstract:** Due to the success of generative flows to model data
distributions, they have been explored in inverse problems.
Given a pre-trained generative flow, previous work pro-
posed to minimize the 2-norm of the latent variables as
a regularization term. The intuition behind it was to en-
sure high likelihood latent variables that produce the clos-
est restoration. However, high-likelihood latent variables
may generate unrealistic samples as we show in our exper-
iments. We therefore propose a solver to directly produce
high-likelihood reconstructions. We hypothesize that our
approach could make generative flows a general purpose
solver for inverse problems. Furthermore, we propose 1 × 1
coupling functions to introduce permutations in a genera-
tive flow. It has the advantage that its inverse does not re-
quire to be calculated in the generation process. Finally, we
evaluate our method for denoising, deblurring, inpainting,
and colorization. We observe a compelling improvement of
our method over prior works.

## System requirements:

* We recommend Linux, but Windows is supported.
* 64-bit Python 3.8 installation.
* Pytorch 1.5.0 or newer with GPU support.
* One NVIDIA GPU with at least 2GB of DRAM.

## Training:

With the default configuration:

```
$ python train.py
```

## Solver:

With the default configuration:

```
$ python solver.py
```

For denoising:

```
$ python solver.py --invprob=denoising
```

