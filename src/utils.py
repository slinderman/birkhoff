import numpy as np
import scipy
from scipy.misc import logsumexp
from scipy.special import gammaln, beta
from scipy.integrate import simps

import scipy.sparse

def logistic(x):
    return 1./(1+np.exp(-x))

def logit(p):
    return np.log(p/(1-p))

def psi_to_pi(psi, axis=None):
    """
    Convert psi to a probability vector pi
    :param psi:     Length K-1 vector
    :return:        Length K normalized probability vector
    """
    if axis is None:
        if psi.ndim == 1:
            K = psi.size + 1
            pi = np.zeros(K)

            # Set pi[1..K-1]
            stick = 1.0
            for k in range(K-1):
                pi[k] = logistic(psi[k]) * stick
                stick -= pi[k]

            # Set the last output
            pi[-1] = stick
            # DEBUG
            assert np.allclose(pi.sum(), 1.0)

        elif psi.ndim == 2:
            M, Km1 = psi.shape
            K = Km1 + 1
            pi = np.zeros((M,K))

            # Set pi[1..K-1]
            stick = np.ones(M)
            for k in range(K-1):
                pi[:,k] = logistic(psi[:,k]) * stick
                stick -= pi[:,k]

            # Set the last output
            pi[:,-1] = stick

            # DEBUG
            assert np.allclose(pi.sum(axis=1), 1.0)

        else:
            raise ValueError("psi must be 1 or 2D")
    else:
        K = psi.shape[axis] + 1
        pi = np.zeros([psi.shape[dim] if dim != axis else K for dim in range(psi.ndim)])
        stick = np.ones(psi.shape[:axis] + psi.shape[axis+1:])
        for k in range(K-1):
            inds = [slice(None) if dim != axis else k for dim in range(psi.ndim)]
            pi[inds] = logistic(psi[inds]) * stick
            stick -= pi[inds]
        pi[[slice(None) if dim != axis else -1 for dim in range(psi.ndim)]] = stick
        assert np.allclose(pi.sum(axis=axis), 1.)

    return pi

def pi_to_psi(pi):
    """
    Convert probability vector pi to a vector psi
    :param pi:      Length K probability vector
    :return:        Length K-1 transformed vector psi
    """
    if pi.ndim == 1:
        K = pi.size
        assert np.allclose(pi.sum(), 1.0)
        psi = np.zeros(K-1)

        stick = 1.0
        for k in range(K-1):
            psi[k] = logit(pi[k] / stick)
            stick -= pi[k]

        # DEBUG
        assert np.allclose(stick, pi[-1])
    elif pi.ndim == 2:
        M, K = pi.shape
        assert np.allclose(pi.sum(axis=1), 1.0)
        psi = np.zeros((M,K-1))

        stick = np.ones(M)
        for k in range(K-1):
            psi[:,k] = logit(pi[:,k] / stick)
            stick -= pi[:,k]
        assert np.allclose(stick, pi[:,-1])
    else:
        raise NotImplementedError

    return psi

def det_jacobian_pi_to_psi(pi):
    """
    Compute |J| = |d\psi_j / d\pi_k| = the jacobian of the mapping from
     pi to psi. Since we have a stick breaking construction, the Jacobian
     is lower triangular and the determinant is simply the product of the
     diagonal. For our model, this can be computed in closed form. See the
     appendix of the draft.

    :param pi: K dimensional probability vector
    :return:
    """
    # import pdb; pdb.set_trace()
    K = pi.size

    # Jacobian is K-1 x K-1
    diag = np.zeros(K-1)
    for k in range(K-1):
        diag[k] = (1.0 - pi[:k].sum()) / (pi[k] * (1-pi[:(k+1)].sum()))

    det_jacobian = diag.prod()
    return det_jacobian

def det_jacobian_psi_to_pi(psi):
    """
    Compute the Jacobian of the inverse mapping psi to pi.
    :param psi:
    :return:
    """
    pi = psi_to_pi(psi)
    return 1.0 / det_jacobian_pi_to_psi(pi)


def gaussian_to_pi_density(psi_mesh, mu, Sigma):
    pi_mesh = np.array(list(map(psi_to_pi, psi_mesh)))
    valid_pi = np.all(np.isfinite(pi_mesh), axis=1)
    pi_mesh = pi_mesh[valid_pi,:]

    # Compute the det of the Jacobian of the inverse mapping
    det_jacobian = np.array(list(map(det_jacobian_pi_to_psi, pi_mesh)))
    det_jacobian = det_jacobian[valid_pi]

    # Compute the multivariate Gaussian density
    # pi_pdf = np.exp(log_dirichlet_density(pi_mesh, alpha=alpha))
    from scipy.stats import multivariate_normal
    psi_dist = multivariate_normal(mu, Sigma)
    psi_pdf = psi_dist.pdf(psi_mesh)
    psi_pdf = psi_pdf[valid_pi]

    # The psi density is scaled by the det of the Jacobian
    pi_pdf = psi_pdf * det_jacobian

    return pi_mesh, pi_pdf

def ln_psi_to_pi(psi):
    """
    Convert the logistic normal psi to a probability vector pi
    :param psi:     Length K vector
    :return:        Length K normalized probability vector
    """
    lognumer = psi

    if psi.ndim == 1:
        logdenom = logsumexp(psi)
    elif psi.ndim == 2:
        logdenom = logsumexp(psi, axis=1)[:, None]
    pi = np.exp(lognumer - logdenom)
    # assert np.allclose(pi.sum(), 1.0)

    return pi

def ln_pi_to_psi(pi, scale=1.0):
    """
    Convert the logistic normal psi to a probability vector pi
    The transformation from psi to pi is not invertible unless
    you know the scaling of the psis.

    :param pi:      Length K vector
    :return:        Length K unnormalized real vector
    """
    assert scale > 0
    if pi.ndim == 1:
        assert np.allclose(pi.sum(), 1.0)
    elif pi.ndim == 2:
        assert np.allclose(pi.sum(1), 1.0)

    psi = np.log(pi) + np.log(scale)

    # assert np.allclose(pi, ln_psi_to_pi(psi))
    return psi

def compute_uniform_mean_psi(K, alpha=2):
    """
    Compute the multivariate distribution over psi that will yield approximately
    Dirichlet(\alpha) prior over pi

    :param K:   Number of entries in pi
    :return:    A K-1 vector mu that yields approximately uniform distribution over pi
    """
    mu, sigma = compute_psi_cmoments(alpha*np.ones(K))
    return mu, np.diag(sigma)

def compute_psi_cmoments(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    mu = np.zeros(K-1)
    sigma = np.zeros(K-1)
    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        mu[k] = simps(psi*density(psi),psi)
        sigma[k] = simps(psi**2*density(psi),psi) - mu[k]**2
        # print '%d: mean=%0.3f var=%0.3f' % (k, mean, s - mean**2)

    return mu, sigma

def get_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / scipy.special.beta(alpha_k,alpha_rest)
    return density

def plot_psi_marginals(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    import matplotlib.pyplot as plt
    plt.figure()

    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        plt.subplot(2,1,1)
        plt.plot(psi,density(psi),label='psi_%d' % k)
        plt.subplot(2,1,2)
        plt.plot(psi,np.log(density(psi)),label='psi_%d' % k)
    plt.subplot(2,1,1)
    plt.legend()

def N_vec(x, axis=None):
    """
    Compute the count vector for PG Multinomial inference
    :param x:
    :return:
    """
    if axis is None:
        if x.ndim == 1:
            N = x.sum()
            return np.concatenate(([N], N - np.cumsum(x)[:-2]))
        elif x.ndim == 2:
            N = x.sum(axis=1)
            return np.hstack((N[:,None], N[:,None] - np.cumsum(x, axis=1)[:,:-2]))
        else:
            raise ValueError("x must be 1 or 2D")
    else:
        inds = [slice(None) if dim != axis else None for dim in range(x.ndim)]
        inds2 = [slice(None) if dim != axis else slice(None,-2) for dim in range(x.ndim)]
        N = x.sum(axis=axis)
        return np.concatenate((N[inds], N[inds] - np.cumsum(x,axis=axis)[inds2]), axis=axis)

def kappa_vec(x, axis=None):
    """
    Compute the kappa vector for PG Multinomial inference
    :param x:
    :return:
    """
    if axis is None:
        if x.ndim == 1:
            return x[:-1] - N_vec(x)/2.0
        elif x.ndim == 2:
            return x[:,:-1] - N_vec(x)/2.0
        else:
            raise ValueError("x must be 1 or 2D")
    else:
        inds = [slice(None) if dim != axis else slice(None,-1) for dim in range(x.ndim)]
        return x[inds] - N_vec(x, axis)/2.0

# is this doing overlapping work with dirichlet_to_psi_density_closed_form?
def get_marginal_psi_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / beta(alpha_k,alpha_rest)
    return density


def dirichlet_to_psi_meanvar(alphas,psigrid=np.linspace(-10,10,1000)):
    K = alphas.shape[0]

    def meanvar(k):
        density = get_marginal_psi_density(alphas[k], alphas[k+1:].sum())
        mean = simps(psigrid*density(psigrid),psigrid)
        s = simps(psigrid**2*density(psigrid),psigrid)
        return mean, s - mean**2

    return list(map(np.array, list(zip(*[meanvar(k) for k in range(K-1)]))))
