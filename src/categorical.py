import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian

import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("white")
seaborn.set_context("talk")

def sigmoid(x): return 1 / (1 + np.exp(-x))

def logit(pi): return np.log(pi / (1-pi))

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[ :D], params[ D:]
    return mean, log_std

def unpack_kumaraswamy_params(params):
    D = np.shape(params)[-1] // 2
    a, b = params[:D], params[D:]
    return a,b

def unpack_gumbell_params(params):
    a=params
    return a

def sample_diag_gumbell(a,temp):
    return (np.log(a)-np.log(-np.log(np.random.uniform(0, 1, size=(a.shape[0])))))/temp

def diag_gumbell(params,temp):
    a=unpack_gumbell_params(params)
    return sample_diag_gumbell(a,temp)


def sample_diag_kumaraswamy(a,b):
    #seed = nr
    return np.power(1-np.power(1-np.random.uniform(low=0.0, high=1.0, size=a.shape[0]),np.power(a,-1)),np.power(b,-1))

def diag_kumaraswamy(params):
    a, b = unpack_kumaraswamy_params(params)
    return sample_diag_kumaraswamy(a, b)

# Categorical distributions
def psi_to_pi(psi):
    pi = []
    # We could also write this with loops
    for i in range(psi.shape[0]):
        pi.append(psi[i] * (1 - np.sum(np.array(pi))))
    pi.append(1-np.sum(np.array(pi)))
    return np.array(pi)

def sample_pi_gaussian(params, noise, temp):
    mean, log_std = unpack_gaussian_params(params)
    sample = noise * np.exp(log_std) / temp + mean
    psi = sigmoid(sample)
    return psi_to_pi(psi)

def sample_pi_kumaraswamy(params, temp):
    psi = diag_kumaraswamy(params)
    return psi_to_pi(psi)

def sample_pi_gumbell(params, temp):
    sample = diag_gumbell(params,temp)
    return np.exp(sample)/np.sum(np.exp(sample))

### Computing the density of p(pi | params)
def density_pi_gaussian(pi, params, temp):
    assert pi.ndim == 1
    K = pi.shape[0]
    mean, log_std = unpack_gaussian_params(params)
    std = np.exp(log_std) / temp

    p_pi = 0
    # We could also write this with loops
    for i in range(K-1):
        ub = 1 - np.sum(pi[:i])

        # Computing the determinant of the inverse tranformation
        p_pi *= ub / (pi[i] * (ub - pi[i]))

        # Compute p(psi[i] | mu, sigma)
        psi_i = logit(pi[i] / ub)
        p_pi *= 1./np.sqrt(2 * np.pi * std[i]**2) \
                * np.exp(-0.5 * (psi_i - mean[i])**2 / std[i]**2)

    return p_pi

def log_density_pi_gaussian(pi, params, temp):
    assert pi.ndim == 1
    K = pi.shape[0]
    mean, log_std = unpack_gaussian_params(params)
    std = np.exp(log_std) / temp

    log_p = 0
    # We could also write this with loops
    for i in range(K-1):
        ub = 1 - np.sum(pi[:i])

        # Computing the determinant of the inverse tranformation
        log_p += np.log(ub) - np.log(pi[i]) - np.log(ub - pi[i])

        # Compute p(psi[i] | mu, sigma)
        psi_i = logit(pi[i] / ub)
        log_p += -0.5 * np.log(2 * np.pi) - np.log(std[i]) \
                 -0.5 * (psi_i - mean[i])**2 / std[i]**2

    return log_p

def log_probability_gmm(x, pi, model_params):
    N, D = x.shape
    ll = 0
    for n in range(N):
        ll += local_log_probability_gmm(x[n], pi[n], model_params)
    return ll

def local_log_probability_gmm(x_n, pi_n, model_params):
    assert x_n.ndim == 1
    D = x_n.shape[0]
    mu, sigma, alpha = model_params

    E_x = np.dot(pi_n, mu.T)
    ll  = -0.5 * np.log(2 * np.pi * sigma**2) *  D
    ll += -0.5 * np.sum((x_n - E_x)**2 / sigma**2)
    ll += np.sum(np.log(np.dot(pi_n, alpha)))
    return ll

def elbo_gmm(x, var_params, model_params, epsilon, temp):
    elbo = 0

    assert epsilon.ndim == 3
    N_samples = epsilon.shape[0]
    N, D = x.shape

    for s in range(N_samples):
        for n in range(N):
            pi_n = sample_pi_gaussian(var_params, epsilon[s, n], temp)
            elbo += local_log_probability_gmm(x[n], pi_n, model_params)
            elbo -= log_density_pi_gaussian(pi_n, var_params, temp)

    elbo /= N_samples

    return elbo

def doublystochastic_breaking(Psi):
    """
    Same as above but with a sample on the interval (lb, ub)
    """
    N = Psi.shape[0] + 1

    P=[]
    #aux = np.zeros((N , N ))
    for i in range(N - 1):

        P.append([])
        #P[0].append(1)


        for j in range(N - 1):

            # Upper bounded by partial row sum
            ub_row = 1 - sum(P[i][:j])

            # Upper bounded by partial column sum

            ub_col = 1 - sum([row[j] for row in P[:i]])


            # Lower bounded (see notes)

            #lb_rem = (1 - sum(P[i][:j])) - (N - (j + 1)) + np.sum(aux[:i,j+1:])
            lb_rem = (1 - sum(P[i][:j])) - (N - (j + 1)) + np.sum([k for row in P[:i] for k in row[j+1:]])
            #print (i,j,P[:i])
            #print (i,j,[k for row in P[:i] for k in row[j+1:]])
            #print (i,j,aux[:i,j+1:])

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            # Sample

            P[i].append( lb + (ub - lb) * sigmoid(Psi[i, j]))
            #aux[i,j]= lb + (ub - lb) * sigmoid(Psi[i, j])

        # Finish off the row
        P[i].append(1 - np.sum(P[i][:-1]))
        #aux[i,-1]= 1 - np.sum(P[i][:-1])
    # Finish off the columns
    P.append([])
    for j in range(N):
        P[N-1].append(1 - np.sum([row[j] for row in P[:N-1]]))
    return np.array([element for row in P for element in row])

def sample_doubly_stochastic(Psi, verbose=False):
    """
    Same as above but with a sample on the interval (lb, ub)
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

            # Sample
            P[i, j] = lb + (ub - lb) * sigmoid(Psi[i, j])

        # Finish off the row
        P[i, -1] = 1 - np.sum(P[i, :-1])

    # Finish off the columns
    for j in range(N):
        P[-1, j] = 1 - np.sum(P[:-1, j])

    return P

    #return [item for sublist in P for item in sublist]

K = 4
mu = np.zeros(K-1)
logsigma = np.zeros(K-1)
epsilon = npr.randn(K - 1)
var_params = np.concatenate((mu, logsigma))
pi = sample_pi_gaussian(var_params, epsilon, 1.0)

print(log_density_pi_gaussian(pi, var_params, 10.0))
print(np.sum(-0.5 * np.log(2*np.pi) - 0.5 * epsilon ** 2))


# plt.bar(np.arange(K), pi)
# plt.show()

# jac = jacobian(sample_pi_gaussian, argnum=0)
# plt.imshow(jac(params, noise, 1.0), interpolation="none")
# plt.colorbar()

g = grad(log_density_pi_gaussian, argnum=1)
print(g(pi, var_params, 1000.0))


# Simulate some data
N = 1
D = 2

# Set mixture model parameters
mu = npr.randn(D, K)
sigma = 0.1
alpha = np.ones(K)/K
model_params = (mu, sigma, alpha)

# Set latent variables
i_true = npr.randint(K, size=N)
z_true = np.zeros((N,K))
z_true[np.arange(N), i_true] = 1

# Generate noisy data
x_true = z_true.dot(mu.T)
x_true += sigma * npr.randn(N, D)
#print log_probability_gmm(x_true, z_true, model_params)

# Monte Carlo estimate of the ELBO
N_samples = 10
epsilon = npr.randn(N_samples, N, K - 1)
elbo_gmm(x_true, var_params, model_params, epsilon, temp=1)

# Compute gradient of the ELBO wrt var_params
g_elbo = grad(elbo_gmm, argnum=1)

# Stochastic gradient ascent of the ELBO
n_iter = 200
stepsize = 0.01
elbo_iterations= np.zeros(n_iter)
for n in range(n_iter):
    if n % 10 == 0:
        print("Iteration ", n)
    epsilon = npr.randn(N_samples, N, K - 1)
    var_params += stepsize*g_elbo(x_true, var_params, model_params, epsilon, temp=1)
    elbo_iterations[n] = elbo_gmm(x_true, var_params, model_params, epsilon, temp=1)


pi_samples = np.array([sample_pi_gaussian(var_params, npr.randn(K - 1), temp=1)
                       for _ in range(N_samples)])

posterior = np.zeros(K)
for k in range(K):
    ek = np.zeros(K)
    ek[k] = 1
    posterior[k] = local_log_probability_gmm(x_true[0], ek, model_params)

from scipy.misc import logsumexp
posterior = np.exp(posterior - logsumexp(posterior))

print("posterior: ", np.round(posterior, 3))
print("var. mean: ", np.round(pi_samples.mean(axis=0), 3))

plt.plot(elbo_iterations)
plt.show()