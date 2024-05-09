##  Variational Inference for SDEs Driven by Fractional Noise

This codebase is structured as a Python package. First [install JAX](https://github.com/google/jax#installation) (CPU or GPU), then run `pip install -e .`.
If you run code that needs the PyTorch dataloader, we suggest installing the CPU version of PyTorch, due to possible CUDA version mismatch with JAX.

Some code parts that might be of particular interest:
 - Run stochastic moving MNIST experiment: sde/jax/train.py
 - Run fOU bridge experiments: experiments/bridge/main.py
 - Implementation of our method in Diffrax: sde/jax/markov_approximation.py / solve_diffrax()
 - Simple implementation of a Euler solver for our method: sde/jax/markov_approximation.py / solve_vector()
 - Implementation of our SDE model driven by MA-fBM: sde/jax/models / FractionalSDE()
 - Implementation of our latent SDE video model driven by MA-fBM: sde/jax/models / VideoSDE()
 - Implementation of optimized omega values: sde/jax/markov_approximation.py / omega_optimized_1(), omega_optimized_2() (type 1 and 2 respectively)
 - Numerically stable implementation of Q(z,x)e^x: sde/jax/markov_approximation.py / gammaincc_ez()
