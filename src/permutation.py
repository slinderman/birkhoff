import sys
import signal
import warnings

import matplotlib.pyplot as plt
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

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
from autograd.optimizers import adam

npr.seed(0)

# Set a global tolerance...
TOL = 1e-16

# First consider sampling a random doubly stochastic matrix
def logistic(psi):
    return 1. / (1 + np.exp(-psi))

def logit(p):
    return np.log(p / (1-p))

def dlogit(p):
    """
    d/dp log(p) - log(1-p)
    = 1/p + 1/(1-p)
    = (1-p + p) / (p * (1-p))
     = (1 / p) * (1 / (1 - p))
    :param p:
    :return:
    """
    return (1.0 / p) * (1.0 / (1.0 - p))

def log_dlogit(p):
    return -np.log(p) - np.log(1-p)

def gaussian_logp(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma**2) -0.5 * (x - mu)**2 / sigma**2

def psi_to_pi_assignment(Psi, tol=TOL, verbose=False):
    """
    Transform a (K-1) x (K-1) matrix Psi into a KxK doubly
    stochastic matrix Pi.  I've introduced some tolerance to
    make it more stable, but I'm not convinced of this approach yet.

    Note that this version uses direct assignment to the matrix P,
    but this is not amenable to automatic differentiation with
    autograd. The next version below uses list comprehension to
    perform the same psi -> pi conversion in an automatically
    differentiable way.
    """
    N = Psi.shape[0] + 1
    assert Psi.shape == (N - 1, N - 1)

    P = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        for j in range(N - 1):
            # Upper bounded by partial row sum
            ub_row = 1 - np.sum(P[i, :j])

            # Upper bounded by partial column sum
            ub_col = 1 - np.sum(P[:i, j])

            # Lower bounded (see notes)
            lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # Check if difference is less than allowable tolerance
            if ub - lb < tol:
                P[i, j] = tol * logistic(Psi[i, j])
            else:
                P[i, j] = lb + (ub - lb) * logistic(Psi[i, j])

        # Finish off the row
        P[i, -1] = 1 - np.sum(P[i, :-1])

    # Finish off the columns
    for j in range(N):
        P[-1, j] = 1 - np.sum(P[:-1, j])

    return P

def psi_to_pi_list(Psi, tol=TOL, verbose=False):
    """
    Autograd doesn't work with array assignment.  Rewrite with
    list comprehension instead.  This will be a bit slower, but
    not too bad.
    """
    N = Psi.shape[0] + 1
    assert Psi.shape == (N - 1, N - 1)

    # P = np.zeros((N, N), dtype=float)
    P = []
    for i in range(N - 1):
        P_i = []
        for j in range(N - 1):
            # Upper bounded by partial row sum
            ub_row = 1.0
            for jj in range(j):
                ub_row = ub_row - P_i[jj]

            # Upper bounded by partial column sum
            # ub_col = 1 - np.sum(P[:i, j])
            ub_col = 1.0
            for ii in range(i):
                ub_col = ub_col - P[ii][j]

            # Lower bounded (see notes)
            # lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])
            lb_rem = 1.0 - (N - (j + 1))
            for jj in range(j):
                lb_rem = lb_rem - P_i[jj]

            for ii in range(i):
                for jj in range(j+1,N):
                    lb_rem = lb_rem + P[ii][jj]

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # Check if difference is less than allowable tolerance
            if ub - lb < tol:
                # P[i, j] = tol * logistic(Psi[i, j])
                P_i.append(tol * logistic(Psi[i, j]))
                warnings.warn("psi_to_pi: bounds less than tol!")
            else:
                # P[i, j] = lb + (ub - lb) * logistic(Psi[i, j])
                P_i.append(lb + (ub - lb) * logistic(Psi[i, j]))

        # Finish off the row
        # P[i, -1] = 1 - np.sum(P[i, :-1])
        PiN = 1.0
        for jj in range(N-1):
            PiN = PiN - P_i[jj]
        P_i.append(PiN)

        # Append
        P.append(P_i)

    # Finish off the columns
    P_N = []
    for j in range(N):
        PjN = 1.0
        for ii in range(N-1):
            PjN = PjN - P[ii][j]
        P_N.append(PjN)
    P.append(P_N)

    return np.array(P)

def check_doubly_stochastic_stable(P, tol=TOL):
    N = P.shape[0]
    assert np.allclose(P.sum(0), 1, atol=N * tol)
    assert np.allclose(P.sum(1), 1, atol=N * tol)
    assert np.min(P) >= -N * tol
    assert np.max(P) <= 1 + N * tol
    print("P min: {0:.5f} max {1:.5f}".format(P.min(), P.max()))

### Compute the density of p(P | mu, sigma)
def pi_to_psi_list(P, tol=TOL, verbose=False):
    """
    Invert Pi to get Psi, assuming Pi was sampled using
    sample_doubly_stochastic_stable() with the same tolerance.
    """
    N = P.shape[0] + 1
    assert P.shape == (N-1, N-1)

    # Psi = np.zeros((N - 1, N - 1), dtype=float)
    Psi = []
    for i in range(N - 1):
        Psi_i = []
        for j in range(N - 1):
            # Upper bounded by partial row sum
            ub_row = 1 - np.sum(P[i, :j])
            # Upper bounded by partial column sum
            ub_col = 1 - np.sum(P[:i, j])

            # Lower bounded (see notes)
            lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # Check if difference is less than allowable tolerance
            if ub - lb < tol:
                # Psi[i, j] = logit(P[i, j] / tol)
                warnings.warn("pi_to_psi: bounds less than tol!")
                Psi_i.append(logit(P[i, j] / tol))
            else:
                # Psi[i, j] = logit((P[i, j] - lb) / (ub - lb))
                Psi_i.append(logit((P[i, j] - lb) / (ub - lb)))

            # assert np.isfinite(Psi[i, j])

        Psi.append(Psi_i)
    return np.array(Psi)


def log_det_jacobian(P, tol=TOL, verbose=False):
    """
    Compute log det of the jacobian of the inverse transformation.
    Evaluate it at the given value of Pi.
    :param P: Given permutation matrix
    :return: |dPsi / dPi |
    """
    N = P.shape[0]
    assert P.shape == (N, N)

    logdet = 0
    for i in range(N - 1):
        for j in range(N - 1):
            # Upper bounded by partial row sum
            ub_row = 1 - np.sum(P[i, :j])
            # Upper bounded by partial column sum
            ub_col = 1 - np.sum(P[:i, j])

            # Lower bounded (see notes)
            lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # Check if difference is less than allowable tolerance
            if ub - lb < tol:
                warnings.warn("log_det_jacobian: bounds less than tol!")
                logdet = logdet + log_dlogit(P[i, j] / tol)
            else:
                logdet = logdet + log_dlogit((P[i, j] - lb) / (ub - lb))
                logdet = logdet - np.log(ub - lb)

    return logdet

def log_density_pi(P, mu, sigma, tol=TOL, verbose=False):
    """
    Compute the log probability of a permutation matrix for a given
    mu and sigma---the parameters of the Gaussian prior on psi.
    """
    N = P.shape[0]
    assert P.shape == (N, N)
    assert mu.shape == (N-1, N-1)
    assert sigma.shape == (N-1, N-1)

    Psi = pi_to_psi_list(P[:-1, :-1], tol=tol, verbose=verbose)
    return log_det_jacobian(P, tol=tol, verbose=verbose) + \
           np.sum(gaussian_logp(Psi, mu, sigma))

### Simple tests/debugging helpers
def sanity_check():
    """
    Make sure everything runs
    """
    K = 4
    mu = np.zeros((K - 1, K - 1))
    sigma = 0.1 * np.ones((K - 1, K - 1))
    Psi = mu + sigma * np.random.randn(K - 1, K - 1)

    # Convert Psi to P via stick breaking raster scan
    P = psi_to_pi_list(Psi)
    check_doubly_stochastic_stable(P)
    log_density_pi(P, mu, sigma)

    test_log_det_jacobian(P)
    test_plot_jacobian(P)

def test_log_det_jacobian(P, tol=TOL, verbose=False):
    """
    Test the log det Jacobian calculation with Autograd
    """
    N = P.shape[0]
    jac = jacobian(pi_to_psi_list)(P[:-1, :-1])
    jac = jac.reshape(((N - 1) ** 2, (N - 1) ** 2))
    sign, logdet_ag = np.linalg.slogdet(jac)
    assert sign == 1.0

    logdet_man = log_det_jacobian(P, tol=tol, verbose=verbose)

    print("log det autograd: ", logdet_ag)
    print("log det manual:   ", logdet_man)
    assert np.allclose(logdet_ag, logdet_man, atol=1e0)

def test_plot_jacobian(P):
    jac = jacobian(pi_to_psi_list)
    J = jac(P[:-1,:-1])
    J = J.reshape(((K - 1) ** 2, (K - 1) ** 2))

    plt.imshow(J, interpolation="none")
    plt.xlabel("$\\mathrm{vec}(\\Pi_{1:K-1,1:K-1})$")
    plt.ylabel("$\\mathrm{vec}(\\Psi)$")
    plt.title("Lower triangular Jacobian")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # Test log det calculations
    # sanity_check()

    ### Set up a simple matching problem
    K = 10
    D = 2
    eta = 0.2
    mus = 2 * npr.randn(K, D)
    num_mcmc_samples = 10

    # Sample a true permutation (in=col, out=row)
    P_true = np.zeros((K, K))
    P_true[np.arange(K), npr.permutation(K)] = 1

    # Sample data according to this permutation
    mus_perm = P_true.dot(mus)
    xs = mus_perm + eta * npr.randn(K, D)

    # Build variational objective.
    def unpack_params(params):
        # Variational dist is a diagonal Gaussian over the (K-1)**2 parameters
        assert params.shape == (2 * (K - 1)**2, )
        mu = np.reshape((params[:(K-1)**2]), (K-1, K-1))
        log_sigma = np.reshape((params[(K-1)**2:]), (K-1, K-1))
        sigma = np.exp(log_sigma)
        return mu, log_sigma, sigma

    # Set up the log probability objective
    # Assume a uniform prior on P?
    # Right now this is just the likelihood...
    def log_prob(P, t):
        return np.sum(gaussian_logp(xs, np.dot(P, mus), eta))

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mu, log_sigma, sigma = unpack_params(params)
        Psi_samples = mu + npr.randn(num_mcmc_samples, K-1, K-1) * sigma
        P_samples = [psi_to_pi_list(Psi) for Psi in Psi_samples]

        elbo = 0
        # 1. Ignore the entropy term
        # for P, Psi in zip(P_samples, Psi_samples):
        #     elbo = elbo + log_prob(P, t) / num_mcmc_samples


        # 2. Do the inverse transformation
        # for P, Psi in zip(P_samples, Psi_samples):
        #     elbo = elbo + (log_prob(P, t) - log_density_pi(P, mu, sigma)) / num_mcmc_samples

        # 3. Use Psi to prevent inverse
        # for P, Psi in zip(P_samples, Psi_samples):
        #     elbo = elbo + log_prob(P, t) / num_mcmc_samples
        #     elbo = elbo - log_det_jacobian(P) / num_mcmc_samples
        #     elbo = elbo - np.sum(gaussian_logp(Psi, mu, sigma)) / num_mcmc_samples

        # 4. Explicitly compute gaussian entropy
        for P, Psi in zip(P_samples, Psi_samples):
            elbo = elbo + log_prob(P, t) / num_mcmc_samples
            elbo = elbo - log_det_jacobian(P) / num_mcmc_samples
        elbo = elbo + 0.5 * (K-1)**2 * (1.0 + np.log(2 * np.pi)) + np.sum(log_sigma)

        # Minimize the negative elbo
        return -elbo

    gradient = grad(variational_objective)
    elbos = []

    ### Plotting
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(121, frameon=True)
    ax2 = fig.add_subplot(122, frameon=True)
    plt.ion()
    plt.show(block=False)

    def plot_permutation(ax1, ax2, P):
        ax1.imshow(P_true, interpolation="none", vmin=0, vmax=1)
        ax1.set_title("True")
        ax2.imshow(P, interpolation="none", vmin=0, vmax=1)
        ax2.set_title("Inferred")


    def callback(params, t, g):
        elbos.append(-variational_objective(params, t))
        print("Iteration {} lower bound {}".format(t, elbos[-1]))

        plt.cla()
        mu, log_sigma, sigma = unpack_params(params)
        Psi = mu + sigma * npr.randn(K - 1, K - 1)
        P = psi_to_pi_list(Psi)
        # plot_matching(ax, P)
        plot_permutation(ax1, ax2, P)
        plt.draw()
        plt.pause(1.0 / 30.0)

        # DEBUG -- check if the variance is going to zero
        print("sigma min: ", sigma.min(), "\t sigma max: ", sigma.max())

        if ctrlc_pressed[0]:
            sys.exit()

    # Check for quit
    ctrlc_pressed = [False]
    def ctrlc_handler(signal, frame):
        print("Halting due to Ctrl-C")
        ctrlc_pressed[0] = True
    signal.signal(signal.SIGINT, ctrlc_handler)

    print("Variational inference for matching...")
    init_mean = pi_to_psi_list(1./K * np.ones((K-1, K-1))).ravel()
    init_log_std = np.zeros((K - 1) ** 2)
    init_var_params = np.concatenate([init_mean, init_log_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=100, callback=callback)

    # Plot the elbo
    plt.figure()
    plt.plot(elbos)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")

    # Sample from the posterior and show samples
    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    def plot_matching(ax, P, xlimits=[-5, 5], ylimits=[-5, 5]):
        # Plot the true mus and the inferred labels
        inv_matching = np.argmax(P, axis=0)  # which datapoint gets output k
        for k in range(K):
            plt.plot(xs[k, 0], xs[k, 1], 'sk', markersize=10)
            plt.plot(mus[k, 0], mus[k, 1], 'ok', markersize=10)

        for k in range(K):
            plt.plot(mus[k, 0], mus[k, 1], 'o',
                     color=colors[k],  markersize=8)
            plt.plot(xs[inv_matching[k], 0], xs[inv_matching[k], 1], 's',
                     markersize=8, color=colors[k])

        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)

    N_samples = 50
    mu_post, log_sigma_post ,sigma_post= unpack_params(variational_params)
    Psi_samples = mu_post + npr.randn(N_samples, K - 1, K - 1) * sigma_post
    P_samples = [psi_to_pi_list(Psi) for Psi in Psi_samples]

    for s in range(N_samples):
        plot_matching(ax, P_samples[s])
        ax.set_title("Sample {}".format(s))
        plt.pause(0.01)

