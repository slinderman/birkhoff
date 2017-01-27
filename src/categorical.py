import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
from scipy.misc import logsumexp
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("white")
seaborn.set_context("talk")

def sigmoid(x): return 1 / (1 + np.exp(-x))

def logit(pi): return np.log(pi / (1-pi))

def pack_gaussian_params(mean, log_std):
    return np.concatenate((np.reshape(mean, mean.shape[0] * mean.shape[1]), np.reshape(log_std, log_std.shape[0] * log_std.shape[1])))

def pack_gumbell_params(loga):
    return np.reshape(loga, loga.shape[0] * loga.shape[1])
def pack_kumaraswamy_params(loga,logb):
    return np.concatenate((np.reshape(loga, loga.shape[0] * loga.shape[1]), np.reshape(logb, logb.shape[0] * logb.shape[1])))


def unpack_gumbell_params(params, N):
    loga = params
    return np.reshape(loga, (N, loga.shape[1]/N ) )

def unpack_gaussian_params(params, N):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    means, log_stds = params[ :D], params[ D:]
    return np.reshape(means, (N, len(means)/N)), np.reshape(log_stds, (N, len(log_stds)/N))

def sample_pi_gaussian(params, noise, temp):
    mean, log_std = unpack_gaussian_params(params, noise.shape[0])
    sample = noise * np.exp(log_std) / temp + mean
    psi = sigmoid(sample)
    return psi_to_pi(psi)

def unpack_kumaraswamy_params(params, N):
    # Params of a kumaraswamy.

    D = np.shape(params)[-1] // 2
    loga, logb = params[:D], params[D:]

    return np.reshape(loga, (N, len(loga) / N)), np.reshape(logb, (N, len(logb)/N))


def sample_pi_gumbell(params, noise, temp):
    loga = params
    sample = (loga-np.log(-np.log(noise)))/temp
    return np.exp(sample)/np.sum(np.exp(sample))


def sample_pi_kumaraswamy(params,noise,temp):
    N = noise.shape[0]

    loga, logb = unpack_kumaraswamy_params(params, N)
    psi = np.power(1-np.power(1-noise, np.exp(-logb)), np.exp(-loga))

    return psi_to_pi(psi)

# Categorical distributions
def psi_to_pi(psi):
    pi = []
    # We could also write this with loops

    for n in range(psi.shape[0]):
        pi.append([])
        for i in range(psi.shape[1]):
            pi[n].append(psi[n,i] * (1 - np.sum(np.array(pi[n]),0)))
        pi[n].append(1-np.sum(np.array(pi[n])))
    return np.array(pi)



### Computing the density of p(pi | params)
def density_pi_gaussian(pi, params, temp):

    K = pi.shape[1]
    N = pi.shape[0]
    mean, log_std = unpack_gaussian_params(params, N)
    std = np.exp(log_std) / temp

    p_pi = []

    # We could also write this with loops
    for n in range(N):
        p_pi.append(0)
        for k in range(K-1):
            ub = 1 - np.sum(pi[n,:k])

            # Computing the determinant of the inverse tranformation
            p_pi[n] *= ub / (pi[n,k] * (ub - pi[n,k]))
            # Compute p(psi[i] | mu, sigma)
            psi_i = logit(pi[n,k] / ub)
            p_pi[n] *= 1./np.sqrt(2 * np.pi * std[n,k]**2) \
                    * np.exp(-0.5 * (psi_i - mean[n,k])**2 / std[n,k]**2)
    return p_pi

def log_density_pi_concrete(pi, params, temp):

    log_pi = []
    K = pi.shape[1]
    N = pi.shape[0]

    loga = unpack_gumbell_params(params, N)

    for n in range(N):
        log_pi.append(0)
        sum = 0
        log_pi[n] += (K-1) * np.log(temp) + gammaln(K)

        for k in range(K):
            log_pi[n] += loga[n,k] - (temp + 1) * np.log(pi[n,k])
            sum += np.exp(loga[n,k])* np.power(pi[n,k], -temp)

        log_pi[n] += -np.log(sum)

    return np.array(log_pi)

def log_density_pi_gaussian(pi, params, temp):
    K = pi.shape[1]
    mean, log_std = unpack_gaussian_params(params, pi.shape[0])
    std = np.exp(log_std) / temp
    logp_pi = []
    # We could also write this with loops
    for n in range(mean.shape[0]):
        logp_pi.append(0)
        for i in range(K - 1):
            ub = 1 - np.sum(pi[n, :i])
            # Computing the determinant of the inverse tranformation
            logp_pi[n] += np.log(ub) - np.log(pi[n,i]) - np.log(ub - pi[n,i])
            # Compute p(psi[i] | mu, sigma)
            psi_i = logit(pi[n, i] / ub)

            logp_pi[n] += -0.5 * np.log(2 * np.pi) - np.log(std[n,i]) \
                 -0.5 * (psi_i - mean[n,i])**2 / std[n,i]**2

    return np.array(logp_pi)

def log_density_pi_kumaraswamy(pi, params, temp):
    K = pi.shape[1]
    loga, logb = unpack_kumaraswamy_params(params, pi.shape[0])

    logp_pi = []
    # We could also write this with loops
    for n in range(loga.shape[0]):
        logp_pi.append(0)
        for i in range(K - 1):
            ub = 1 - np.sum(pi[n, :i])
            # Computing the determinant of the inverse tranformation
            logp_pi[n] += -np.log(ub)
            # Compute p(psi[i] | mu, sigma)
            psi_i = pi[n, i] / ub
            logp_pi[n] += loga[n,i] + logb[n,i] + (np.exp(loga[n,i])-1) * np.log(psi_i) + (np.exp(logb[n,i])-1) * np.log(1-np.power(psi_i,np.exp(loga[n,i])))
    return np.array(logp_pi)


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

def elbo_gmm_gaussian(x, var_params, model_params, epsilon, temp):

    elbo = 0
    assert epsilon.ndim == 3
    N_samples = epsilon.shape[0]
    N, D = x.shape

    for s in range(N_samples):
        for n in range(N):
            pi_n = sample_pi_gaussian(var_params, np.reshape(epsilon[s, n],(1,len(epsilon[s,n]))), temp)
            elbo += local_log_probability_gmm(x[n], pi_n, model_params)
            elbo -= log_density_pi_gaussian(pi_n, var_params, temp)

    elbo /= N_samples

    return elbo

def elbo_gmm_gumbell(x, var_params, model_params, epsilon, temp):

    elbo = 0
    assert epsilon.ndim == 3
    N_samples = epsilon.shape[0]
    N, D = x.shape

    for s in range(N_samples):
        for n in range(N):
            pi_n = sample_pi_gumbell(var_params, np.reshape(epsilon[s, n],(1,len(epsilon[s,n]))), temp)
            elbo += local_log_probability_gmm(x[n], pi_n, model_params)
            elbo -= log_density_pi_concrete(pi_n, var_params, temp)

    elbo /= N_samples

    return elbo


def elbo_gmm_kumaraswamy(x, var_params, model_params, epsilon, temp):

    elbo = 0
    assert epsilon.ndim == 3
    N_samples = epsilon.shape[0]
    N, D = x.shape

    for s in range(N_samples):

        for n in range(N):
            pi_n  = sample_pi_kumaraswamy(var_params, np.reshape(epsilon[s, n],(1,len(epsilon[s,n]))), temp)
            elbo += local_log_probability_gmm(x[n], pi_n, model_params)
            elbo -= log_density_pi_kumaraswamy(pi_n, var_params, temp)

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



# Simulate some data
N = 1 #Number of samples
D = 2 #Dimension
K = 7 #Number of classes

# Set mixture model parameters
mu_gmm = npr.randn(D, K)
sigma_gmm = 0.05
alpha_gmm = np.ones(K)/K
model_params = (mu_gmm, sigma_gmm, alpha_gmm)

#set recognition model (initial) params: Gaussian case

mu = np.zeros((N,K-1))
logsigma = np.zeros((N,K-1))
var_params_gaussian = pack_gaussian_params(mu,logsigma)

#set recognition model (initial) params: Gumbell case

loga = np.zeros((N,K))
var_params_gumbell = loga

loga = np.zeros((N,K-1))
logb = np.zeros((N,K-1))
var_params_kumaraswamy = pack_kumaraswamy_params(loga,logb)


# Set latent variables
i_true = npr.randint(K, size=N)
z_true = np.zeros((N,K))
z_true[np.arange(N), i_true] = 1

# Generate noisy data
x_true = z_true.dot(mu_gmm.T)
x_true += sigma_gmm * npr.randn(N, D)
#print log_probability_gmm(x_true, z_true, model_params)

# Monte Carlo number of samples samples (per data point) estimate of the ELBO
N_samples = 10

# Compute gradient of the ELBO wrt var_params
g_elbo_gaussian = grad(elbo_gmm_gaussian, argnum=1)
g_elbo_gumbell = grad(elbo_gmm_gumbell, argnum=1)
g_elbo_kumaraswamy = grad(elbo_gmm_kumaraswamy, argnum=1)

# Stochastic gradient ascent of the ELBO
n_iter = 200
stepsize = 0.01
elbo_iterations= np.zeros((n_iter,3))
tol=0.001
print var_params_gumbell

for n in range(n_iter):
    if n % 10 == 0:
        print("Iteration ", n)
    epsilon_gaussian = npr.randn(N_samples, N, K - 1)
    epsilon_gumbell = npr.uniform(0, 1, (N_samples,N,K))
    epsilon_kumaraswamy = npr.uniform(0.0, 1.0, (N_samples, N, K - 1))

    var_params_gaussian += stepsize*g_elbo_gaussian(x_true, var_params_gaussian, model_params, epsilon_gaussian, temp=1)
    var_params_gumbell += stepsize * g_elbo_gumbell(x_true, var_params_gumbell, model_params, epsilon_gumbell, temp=1)
    var_params_kumaraswamy += stepsize * g_elbo_kumaraswamy(x_true, var_params_kumaraswamy, model_params, epsilon_kumaraswamy, temp=1)

    elbo_iterations[n,0] = elbo_gmm_gaussian(x_true, var_params_gaussian, model_params, epsilon_gaussian, temp=1)
    elbo_iterations[n, 1] = elbo_gmm_gumbell(x_true, var_params_gumbell, model_params, epsilon_gumbell, temp=1)
    elbo_iterations[n, 2] = elbo_gmm_kumaraswamy(x_true, var_params_kumaraswamy, model_params, epsilon_kumaraswamy,temp=1)

    #if(np.abs(elbo_iterations[n,0]-elbo_iterations[n-1])<tol):
     #   break

N_samples_test=100
pi_samples_gaussian = np.reshape(np.asarray([sample_pi_gaussian(var_params_gaussian, npr.randn(N,K - 1), temp=1)
                       for _ in range(N_samples_test)]),(N_samples_test,K))
pi_samples_gumbell = np.reshape(np.asarray([sample_pi_gumbell(var_params_gumbell, npr.uniform(0,1,(N,K)), temp=1)
                       for _ in range(N_samples_test)]),(N_samples_test,K))

pi_samples_kumaraswamy = np.reshape(np.asarray([sample_pi_kumaraswamy(var_params_kumaraswamy, npr.uniform(0,1,(N,K - 1)), temp=1)
                       for _ in range(N_samples_test)]),(N_samples_test,K))

one_hot_gaussian=np.zeros((N_samples_test,K))
one_hot_gaussian[range(N_samples_test),np.argmax(pi_samples_gaussian,axis=1)]=1

one_hot_gumbell=np.zeros((N_samples_test,K))
one_hot_gumbell[range(N_samples_test),np.argmax(pi_samples_gumbell,axis=1)]=1

one_hot_kumaraswamy=np.zeros((N_samples_test,K))
one_hot_kumaraswamy[range(N_samples_test),np.argmax(pi_samples_kumaraswamy,axis=1)]=1

z_post_approx=np.mean(one_hot_gumbell,axis=0)

posterior = np.zeros(K)
for k in range(K):
    ek = np.zeros(K)
    ek[k] = 1
    posterior[k] = local_log_probability_gmm(x_true[0], ek, model_params)


posterior = np.exp(posterior - logsumexp(posterior))

print("posterior : ", np.round(posterior, 3))
print("var. mean (gaussian): ", np.round(pi_samples_gaussian.mean(axis=0), 3))
print("var. round mean (gaussian): ", np.round(np.mean(one_hot_gaussian,axis=0), 3))

print("posterior: ", np.round(posterior, 3))
print("var. mean (gumbell): ", np.round(pi_samples_gumbell.mean(axis=0), 3))
print("var. round mean (gumbell): ", np.round(np.mean(one_hot_gumbell,axis=0), 3))

print("posterior: ", np.round(posterior, 3))
print("var. mean (kumaraswamy): ", np.round(pi_samples_kumaraswamy.mean(axis=0), 3))
print("var. round mean (kumaraswamy): ", np.round(np.mean(one_hot_kumaraswamy,axis=0), 3))


plt.plot(elbo_iterations[:,0],'r')
plt.plot(elbo_iterations[:,1],'b')
plt.plot(elbo_iterations[:,2],'g')
plt.show()


#plt.bar(np.arange(K), pi)
#plt.show()

# jac = jacobian(sample_pi_gaussian, argnum=0)
# plt.imshow(jac(params, noise, 1.0), interpolation="none")
# plt.colorbar()