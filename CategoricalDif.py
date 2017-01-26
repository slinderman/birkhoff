import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from copy import deepcopy
from autograd import jacobian

def sigmoid(x): return 1 / (1 + np.exp(-x))

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

def sample_diag_gaussian(mean, log_std,temp):
    #seed = npr.RandomState(0)
    return np.random.randn(mean.shape[0]) * np.exp(log_std)/temp + mean

def diag_gaussian(params,temp):
    mean, log_std = unpack_gaussian_params(params)
    return sample_diag_gaussian(mean,log_std,temp)



def proportionsBreaking(Psi):
    propstrue =[]
    for i in range(Psi.shape[0]):
        propstrue.append(Psi[i]*(1-np.sum(np.array(propstrue))))
    propstrue.append(1-np.sum(np.array(propstrue)))
    return np.array(propstrue)

def sample_proportions_gaussian(params,temp):
    sample=diag_gaussian(params,temp)
    Psi=sigmoid(sample)
    return proportionsBreaking(Psi)

def sample_proportions_kumaraswamy(params,temp):
    Psi = diag_kumaraswamy(params)
    return proportionsBreaking(Psi)

def sample_proportions_gumbell(params,temp):
    sample = diag_gumbell(params,temp)
    return np.exp(sample)/np.sum(np.exp(sample))


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

#g= jacobian(doublystochastic_breaking)
#g(np.ones((5,5)))
#grad_diag_gaussian=elementwise_grad(diag_gaussian)
proportions1 = lambda y: proportionsKumaraswamy(y, 0.1)
proportions2 = lambda y: proportionsGumbell(y,0.1)
proportions2 = lambda y: proportionsGaussian(y,0.1)

#print proportions2(np.ones(8))

#print g(np.ones(8))
Psi=np.ones((3,3))
P=[[1,2,3,4,5],[1,2,3,4]]


#print sum(Psi[0:0,0])
print doublystochastic_breaking(Psi)
print sample_doubly_stochastic(Psi)
g=jacobian(doublystochastic_breaking)
print g(Psi)

