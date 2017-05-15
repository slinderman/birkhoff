import os, sys
from copy import deepcopy

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import autograd.scipy as scipy
from autograd.scipy.misc import logsumexp
from scipy.optimize import linear_sum_assignment
from autograd.optimizers import adam, sgd
from autograd import grad

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from birkhoff.primitives import gaussian_logp, gaussian_entropy, logistic, logit

import seaborn as sns

sns.set_context("talk")
sns.set_style("white")

color_names = ["red",
               "windows blue",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "pastel purple",
               "mint",
               "salmon"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")


def simulate_data(M, T, N, num_poss_per_neuron,
                  sigmasq_W=0.08, etasq=0.1, rho=0.1):

    # Sample global worm variables
    A = npr.rand(N, N) < rho
    W = np.sqrt(sigmasq_W) * npr.randn(N, N)
    assert np.all(abs(np.linalg.eigvals(A * W)) <= 1.0)

    # Sample permutations for each worm
    Ps = np.zeros((M, N, N))
    for m in range(M):
        # perm[i] = index of neuron i in worm m's neurons
        perm = npr.permutation(N)
        Ps[m, np.arange(N), perm] = 1

    # Make constraint matrices for each worm
    Cs = np.ones((M, N, N), dtype=bool)
    for m in range(M):
        for n in range(N):
            i = np.where(Ps[m, n, :] == 1)[0]
            invalid = npr.choice(np.arange(N), N - num_poss_per_neuron, replace=False)
            for j in invalid:
                if j != i:
                    Cs[m,n,j] = False

    # Sample some data!
    Ys = np.zeros((M, T, N))
    for m in range(M):
        Ys[m,0,:] = np.ones(N)
        Wm = Ps[m].T.dot((W*A).dot(Ps[m]))
        for t in range(1, T):
            mu_mt = np.dot(Wm, Ys[m, t-1, :])
            Ys[m,t,:] = mu_mt + np.sqrt(etasq) * npr.randn(N)

    return A, W, Ps, Cs, Ys


if __name__ == "__main__":
    A, W, Ps, Cs, Ys = simulate_data(5, 1000, 100, 20)
