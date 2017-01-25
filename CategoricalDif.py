import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from copy import deepcopy
from autograd import jacobian

def tanh(x):                 # Define a function
     y = np.exp(-x)
     return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)       # Obtain its gradient function

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2

    mean, log_std = params[ :D], params[ D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std):
    seed = npr.RandomState(0)
    return np.random.randn(mean.shape[0]) * np.exp(log_std) + mean

def diag_gaussian(params):
    mean, log_std = unpack_gaussian_params(params)
    return sample_diag_gaussian(mean,log_std)


def sigmoid(x): return 1 / (1 + np.exp(-x))

def proportions(params):
    sample=diag_gaussian(params)

    logsample=sigmoid(sample)

    propstrue =[]
    print logsample
    for i in range(sample.shape[0]):
        #props[i]=logsample[i]*(1-deepcopy(np.sum(props)))
        #print type(5)
        #print type(logsample)
        #props[i]=logsample[i].astype(float)
        propstrue.append(logsample[i]*(1-np.sum(np.array(propstrue))))
    propstrue.append(1-np.sum(np.array(propstrue)))
    return np.array(propstrue)


def func(x):
   prev = 0
   result = []
   for i in range(0, 3):
      prev = r = x**i + prev
      result.append(r)
   return np.array(result)

gg=jacobian(func)
print gg(3)
def elementwise_grad(fun):                   # A wrapper for broadcasting
     return grad(lambda x: np.sum(fun(x)))

seed = npr.RandomState(0)
grad_diag_gaussian=elementwise_grad(diag_gaussian)
diag_gaussian(np.zeros(8))


g=jacobian(proportions)
print grad_diag_gaussian(np.zeros(8))
print g(np.zeros(8))
print proportions(np.zeros(8))