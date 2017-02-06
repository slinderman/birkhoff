import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
from scipy.misc import logsumexp
from scipy.special import gammaln
from scipy.linalg import norm
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("white")
seaborn.set_context("talk")
import cPickle
from autograd.optimizers import adam,rmsprop
import math

def hellinger(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)

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
    return np.reshape(loga, (N, -1 ) )

def unpack_gaussian_params(params, N):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2

    means, log_stds = params[:,:D], params[:,D:]

    return np.reshape(means, (N, -1)), np.reshape(log_stds, (N, -1))


def sample_pi_gaussian(params, noise, temp):
    mean, log_std = unpack_gaussian_params(params, noise.shape[0])
    sample = noise * np.exp(log_std) / np.sqrt(temp) + mean
    psi = sigmoid(sample)
    return psi_to_pi(psi)

def unpack_kumaraswamy_params(params, N):
    # Params of a kumaraswamy.

    D = np.shape(params)[-1] // 2

    loga, logb = params[:,:D], params[:,D:]

    return np.reshape(loga, (N, -1)), np.reshape(logb, (N, -1))


def sample_pi_gumbell(params, noise, temp):
    loga = params

    sample = (loga-np.log(-np.log(noise)))/temp

    sample =(sample.T- np.amax(sample,axis=1)).T

    return (np.exp(sample.T)/np.sum(np.exp(sample),axis=1)).T


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
    std = np.exp(log_std) / np.sqrt(temp)

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

    std = np.exp(log_std) / np.sqrt(temp)
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
    #print E_x.shape
    ll  = -0.5 * np.log(2 * np.pi * sigma**2) *  D
    ll += -0.5 * np.sum((x_n - E_x)**2 / sigma**2,axis =1)

    #ll += np.sum(np.log(np.dot(pi_n, alpha)))

    return np.sum(ll)

def log_evidence_discrete(x_n,model_params):
    assert x_n.ndim ==1
    mu, sigma, alpha = model_params

    K = mu.shape[1]

    ll  = -0.5 * np.log(2 * np.pi * sigma**2) *  D
    ll += -0.5 * np.sum((x_n - mu.T)**2 / sigma**2, axis=1)

    #ll += np.log(1./K)
    assert ll.shape == (K,)
    return logsumexp(ll)+np.log(1./K)

def log_evidence_cont(x_n,model_params,N_samples):
    assert x_n.ndim ==1
    mu, sigma, alpha = model_params
    K = mu.shape[1]

    epsilon = npr.uniform(0, 1, (N_samples, K))
    pi=np.zeros((N_samples,K))

    for n in range(N_samples):
        pi[n,:]=np.random.dirichlet(np.ones(K))

    ll = -0.5 * np.log(2 * np.pi * sigma ** 2) * D
    ll += -0.5 * np.sum((x_n - mu.dot(pi[:,:].T).T)**2 / sigma**2, axis=1)
    return logsumexp(ll)-np.log(len(ll))

def elbo_gmm_gaussian(x, var_params, model_params, temp,N_samples):

    elbo = 0

    N, D = x.shape
    K= var_params.shape[0]/2+1

    epsilon=npr.randn(N_samples, N, K -1 )

    for n in range(N):
        pi_n = sample_pi_gaussian(np.tile(var_params,(N_samples,1)),np.reshape(epsilon[:,n,:],(N_samples,K-1)), temp)
        elbo += local_log_probability_gmm(x[n], pi_n, model_params) -N_samples*np.log(math.factorial(pi_n.shape[1]-1))
        elbo -= np.sum(log_density_pi_gaussian(pi_n, np.tile(var_params,(N_samples,1)), temp))

    elbo /= N_samples

    return elbo



def elbo_gmm_gumbell(x, var_params, model_params, temp,N_samples):

    elbo = 0


    N, D = x.shape
    epsilon = npr.uniform(0, 1, (N_samples, N, K))

    for n in range(N):
        pi_n = sample_pi_gumbell(np.tile(var_params,(N_samples,1)), np.reshape(epsilon[:,n,:],(N_samples,-1)), temp)
        elbo += local_log_probability_gmm(x[n], pi_n, model_params)  -N_samples*np.log(math.factorial(pi_n.shape[1]-1))
        elbo -= np.sum(log_density_pi_concrete(pi_n, np.tile(var_params,(N_samples,1)), temp))

    elbo /= N_samples

    return elbo

def elbo_gmm_kumaraswamy(x, var_params, model_params, temp,N_samples):

    elbo = 0


    N, D = x.shape
    epsilon = npr.uniform(0.0, 1.0, (N_samples, N, K - 1))
    for n in range(N):
        pi_n  = sample_pi_kumaraswamy(np.tile(var_params,(N_samples,1)),np.reshape(epsilon[:,n,:],(N_samples,-1)), temp)
        elbo += local_log_probability_gmm(x[n], pi_n, model_params)  -np.log(math.factorial(pi_n.shape[1]-1))*N_samples
        elbo -= np.sum(log_density_pi_kumaraswamy(pi_n, np.tile(var_params,(N_samples,1)), temp))

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


#Number of simulations
N_simulations=100 #Number of simulations for a single temperature
temps = np.array([0.5, 1, 5, 10]) #temperatures
N_temps = len(temps) #Number of temperatures
N_iters = 300# Stochastic gradient ascent of the ELBO



# Parameters for data simulation
N = 1 #Number of samples
D = 2 #Dimension
K = 10 #Number of classes


#define arrays to store results
elbo_iterations_kumaraswamy = np.zeros((N_simulations, N_temps,N_iters))
elbo_iterations_gaussian = np.zeros((N_simulations, N_temps, N_iters))
elbo_iterations_gumbell = np.zeros((N_simulations,N_temps, N_iters))
dist_gaussian = np.zeros((N_simulations, N_temps, N_iters))
dist_gumbell = np.zeros((N_simulations, N_temps, N_iters))
dist_kumaraswamy = np.zeros((N_simulations, N_temps, N_iters))

approx_z_gaussian=np.zeros((N_simulations, N_temps, N_iters,K))
approx_z_gumbell =np.zeros((N_simulations, N_temps, N_iters,K))
approx_z_kumaraswamy = np.zeros((N_simulations, N_temps, N_iters,K))

var_params_gaussian_all = np.zeros((N_simulations, N_temps, N_iters, N * (K-1) *2 ))
var_params_gumbell_all = np.zeros((N_simulations, N_temps, N_iters, N * (K) ))
var_params_kumaraswamy_all = np.zeros((N_simulations, N_temps, N_iters, N * (K-1) *2 ))

mu_gmm_all = np.zeros((N_simulations,N_temps, D, K))
x_true_all = np.zeros((N_simulations,N_temps, D, N))
z_true_all = np.zeros((N_simulations,N_temps,  N,K))


posterior_all = np.zeros((N_simulations,N_temps,  K))
log_evidence_all=np.zeros((N_simulations,N_temps))
#simulate data now



for s in range(N_simulations):
    for t in range(N_temps):

        print("Iteration ", (t, s))

        # Set mixture model parameters
        mu_gmm = npr.randn(D, K)
        sigma_gmm = 0.05
        alpha_gmm = np.ones(K)/K
        model_params = (mu_gmm, sigma_gmm, alpha_gmm)

        mu_gmm_all[s,t]=mu_gmm

        # Set latent variables
        i_true = npr.randint(K, size=N)
        z_true = np.zeros((N,K))
        z_true[np.arange(N), i_true] = 1

        z_true_all[s,t]=z_true

        # Generate noisy data
        x_true = z_true.dot(mu_gmm.T)
        x_true += sigma_gmm * npr.randn(N, D)
        x_true_all[s,t]=x_true.T

        log_evidence_all[s,t]=log_evidence_cont(x_true[0],model_params,1000)

        #Compute actual posterior probability

        posterior = np.zeros(K)
        for k in range(K):
            ek = np.zeros((1,K))
            ek[0,k] = 1
            posterior[k] = local_log_probability_gmm(x_true[0], ek, model_params)

        posterior = np.exp(posterior - logsumexp(posterior))
        posterior_all[s,t]=posterior

        #set recognition model (initial) params: Gaussian case

        mu = np.zeros((N,K-1))
        logsigma = np.zeros((N,K-1))
        var_params_gaussian = pack_gaussian_params(mu,logsigma)

        #set recognition model (initial) params: Gumbell case

        loga = np.zeros((N,K))
        var_params_gumbell = pack_gumbell_params(loga)

        loga = np.zeros((N,K-1))
        logb = np.zeros((N,K-1))
        var_params_kumaraswamy = pack_kumaraswamy_params(loga,logb)



        # Monte Carlo number of samples samples (per data point) estimate of the ELBO
        N_samples = 10
        elbo_gaussian = lambda x,t: -1*elbo_gmm_gaussian(x_true,x,model_params,temps[0],N_samples)
        elbo_gumbell = lambda x, t: -1 * elbo_gmm_gumbell(x_true, x, model_params, temps[0],N_samples)
        elbo_kumaraswamy = lambda x, t: -1 * elbo_gmm_kumaraswamy(x_true, x, model_params, temps[0],N_samples)
        #print elbo_gaussian_2(var_params_gaussian,1)
        # Compute gradient of the ELBO wrt var_params
        g_elbo_gaussian = grad(elbo_gmm_gaussian,argnum=1)
        g_elbo_gumbell = grad(elbo_gmm_gumbell, argnum=1)
        g_elbo_kumaraswamy = grad(elbo_gmm_kumaraswamy, argnum=1)

        #Gradient ascent
        stepsize = 0.01

        tol=0.001

        variational_params,varseq = adam(grad(elbo_gaussian), var_params_gaussian, step_size=0.1, num_iters=N_iters)
        var_params_gaussian_all[s, t, :] = varseq
        print 'Gaussian Done'
        variational_params, varseq = adam(grad(elbo_gumbell), var_params_gumbell, step_size=0.1, num_iters=N_iters)
        var_params_gumbell_all[s, t, :] = varseq
        print 'Gumbel Done'
        variational_params, varseq = adam(grad(elbo_kumaraswamy), var_params_kumaraswamy, step_size=0.1, num_iters=N_iters)
        print 'Kuma done'
        var_params_kumaraswamy_all[s, t, :] = varseq
        for i in range(N_iters):


            elbo_iterations_gaussian[s,t,i] = elbo_gmm_gaussian(x_true, var_params_gaussian_all[s,t,i], model_params, temps[t],N_samples)
            elbo_iterations_gumbell[s,t,i] = elbo_gmm_gumbell(x_true, var_params_gumbell_all[s,t,i], model_params,  temps[t],N_samples)
            elbo_iterations_kumaraswamy[s,t,i] = elbo_gmm_kumaraswamy(x_true, var_params_kumaraswamy_all[s,t,i], model_params, temps[t],N_samples)

            N_samples_test=1000
            pi_samples_gaussian = np.reshape(np.asarray([sample_pi_gaussian(np.reshape(var_params_gaussian_all[s,t,i],(1,-1)), npr.randn(N,K - 1), temp=temps[t])
                               for _ in range(N_samples_test)]),(N_samples_test,K))
            pi_samples_gumbell = np.reshape(np.asarray([sample_pi_gumbell(var_params_gumbell_all[s,t,i], npr.uniform(0,1,(N,K)), temp=temps[t])
                               for _ in range(N_samples_test)]),(N_samples_test,K))

            pi_samples_kumaraswamy = np.reshape(np.asarray([sample_pi_kumaraswamy(np.reshape(var_params_kumaraswamy_all[s,t,i],(1,-1)), npr.uniform(0,1,(N,K - 1)), temp=temps[t])
                               for _ in range(N_samples_test)]),(N_samples_test,K))

            one_hot_gaussian=np.zeros((N_samples_test,K))
            one_hot_gaussian[range(N_samples_test),np.argmax(pi_samples_gaussian,axis=1)]=1

            one_hot_gumbell=np.zeros((N_samples_test,K))
            one_hot_gumbell[range(N_samples_test),np.argmax(pi_samples_gumbell,axis=1)]=1

            one_hot_kumaraswamy=np.zeros((N_samples_test,K))
            one_hot_kumaraswamy[range(N_samples_test),np.argmax(pi_samples_kumaraswamy,axis=1)]=1

            approx_z_gaussian[s,t,i,:] = np.mean(one_hot_gaussian,axis=0)
            approx_z_gumbell[s,t,i,:] = np.mean(one_hot_gumbell,axis=0)
            approx_z_kumaraswamy[s,t,i,:] = np.mean(one_hot_kumaraswamy,axis=0)

            dist_gaussian[s,t,i]=max(abs(posterior-np.mean(one_hot_gaussian,axis=0)))
            dist_gumbell[s,t,i]=max(abs(posterior-np.mean(one_hot_gumbell,axis=0)))
            dist_kumaraswamy[s,t,i]=max(abs(posterior-np.mean(one_hot_kumaraswamy,axis=0)))

        dict={'dists':[dist_gaussian,dist_gumbell,dist_kumaraswamy],'var_params':[var_params_gaussian_all,var_params_gumbell_all,var_params_kumaraswamy_all], \
     'elbos':[elbo_iterations_gaussian,elbo_iterations_gumbell,elbo_iterations_kumaraswamy],'x_true_all':x_true_all,'z_true_all':z_true_all,\
      'posteriors':posterior_all,'mu_gmm_all':mu_gmm_all,'log_evidences':log_evidence_all,'approx_z':[approx_z_gaussian,approx_z_gumbell,approx_z_kumaraswamy]}
        cPickle.dump( dict, open( "save_k10.p", "wb" ) )
#
# print("posterior : ", np.round(posterior, 3))
# print("var. mean (gaussian): ", np.round(pi_samples_gaussian.mean(axis=0), 3))
# print("var. round mean (gaussian): ", np.round(np.mean(one_hot_gaussian,axis=0), 3))
#
#
# print("posterior: ", np.round(posterior, 3))
# print("var. mean (gumbell): ", np.round(pi_samples_gumbell.mean(axis=0), 3))
# print("var. round mean (gumbell): ", np.round(np.mean(one_hot_gumbell,axis=0), 3))
#
# print("posterior: ", np.round(posterior, 3))
# print("var. mean (kumaraswamy): ", np.round(pi_samples_kumaraswamy.mean(axis=0), 3))
# print("var. round mean (kumaraswamy): ", np.round(np.mean(one_hot_kumaraswamy,axis=0), 3))


#plt.plot(elbo_iterations_gaussian[:,0,:].T,'r')
#plt.plot(elbo_iterations_gumbell[:,0,:].T,'b')
#plt.plot(elbo_iterations_kumaraswamy[:,0,:].T,'g')
#plt.show()

#plt.plot(elbo_iterations_gaussian[:,1,:].T,'r')
#plt.plot(elbo_iterations_gumbell[:,1,:].T,'b')
#plt.plot(elbo_iterations_kumaraswamy[:,1,:].T,'g')
#plt.show()

#plt.plot(dist_gaussian[:,0,:].T,'r')
#plt.plot(dist_gumbell[:,0,:].T,'b')
#plt.plot(dist_kumaraswamy[:,0,:].T,'g')

plt.show()

#plt.plot(dist_gaussian[:,1,:].T,'r')
#plt.plot(dist_gumbell[:,1,:].T,'b')
#plt.plot(dist_kumaraswamy[:,1,:].T,'g')

plt.show()

#plt.show()


#plt.bar(np.arange(K), pi)
#plt.show()

# jac = jacobian(sample_pi_gaussian, argnum=0)
# plt.imshow(jac(params, noise, 1.0), interpolation="none")
# plt.colorbar()