import numpy as np

ndim = 2
labels = ['a', 'b']

# module globals to bypass pickling arguments at every function call
# in parallel execution
xi = None
yi = None


# only a function of theta, use module globals in place of other args
def logPrior(theta):
    return 0


# only a function of theta, use module globals in place of other args
def logLikelihood(theta):
    a, b = theta
    return -np.sum(np.power(yi - (a * xi + b), 2))


# only a function of theta, use module globals in place of other args
def logProbability(theta):
        lp = logPrior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + logLikelihood(theta)
