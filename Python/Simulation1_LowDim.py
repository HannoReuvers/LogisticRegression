import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import os
import re
import time

# Load function library
CurrentDirectory = os.path.dirname(__file__)
ParentDirectory = re.findall('\S+/Python', CurrentDirectory)[0]
LibraryScriptLocation = os.path.join(ParentDirectory, 'functions', 'LogisticRegressionFunctions.py')
exec(open(LibraryScriptLocation).read())

# Parameters
method = 'ScikitLearn'  # Option: 'NewtonRhapsonNumPy' and 'ScikitLearn'
n = 500
b = 0.5
theta = np.array([[0.3], [0.7]])
Nsim = 1000
np.random.seed(1)
bStart = b              # Start optimize from true parameter value
thetaStart = theta      # Start optimize from true parameter value

# Number of features
p = len(theta)

# Initialize matrices
ParaEst = np.empty((p+1, Nsim)); ParaEst[:] = np.NaN
tstat = np.empty((p+1, Nsim)); tstat[:] = np.NaN

#####################
# START MONTE CARLO #
#####################
tic = time.time()
for simiter in range(Nsim):

    # Report progress
    if simiter>0 and simiter%100==0:
        print("Iteration", simiter, "out of", Nsim)

    # Generate logistic regression data set
    X,y = GenerateLogisticData(b, theta, n, rho=0.3)

    # Estimation
    if method=='NewtonRhapsonNumPy':
        bhat, thetahat, _, AsymptoticCovMatrix = EstimateLogisticRegressionNewtonRhapsonNumPy(X, y, bStart, thetaStart)
    elif method=='ScikitLearn':
        bhat, thetahat, AsymptoticCovMatrix = EstimateLogisticRegressionScikitLearn(X.T, y.reshape(n,), True)
        thetahat = thetahat.reshape(p,1)
    # Store results
    ParaEst[:, simiter:(simiter+1)] = np.vstack( (bhat.reshape(1,1),thetahat) )
    tstat[:, simiter:(simiter+1)] = np.sqrt(n)*(ParaEst[:, simiter:(simiter+1)]-np.vstack((np.array([b]), theta)))/np.sqrt(np.diag(AsymptoticCovMatrix).reshape(p+1,1))
toc = time.time()
print("Elapsed time is", toc-tic, "seconds.")

# Report figures
x = np.linspace(-5, 5, 100)
normpdf = norm.pdf(x, 0, 1)
fig, axs = plt.subplots(2, 1)
axs[0].hist(tstat[0,:], bins=20, density=True, alpha=0.6, color='b')
axs[0].plot(x, normpdf, 'k', linewidth=2)
axs[1].hist(tstat[1,:], bins=20, density=True, alpha=0.6, color='b')
axs[1].plot(x, normpdf, 'k', linewidth=2)
plt.show()
