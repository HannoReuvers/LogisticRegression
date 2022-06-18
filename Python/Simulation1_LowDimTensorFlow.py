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
method = 'NewtonRhapsonTensorFlow'
n = 500
b = tf.constant([[0.5]])
theta = tf.constant([[0.3], [0.7]])
Nsim = 1000
np.random.seed(1)
bStart = b              # Start optimize from true parameter value
thetaStart = theta      # Start optimize from true parameter value

# Number of features
p = len(theta)

# Initialize matrices
ParaEst = tf.Variable( tf.zeros((p+1,Nsim)) )
tstat = tf.Variable( tf.zeros((p+1,Nsim)) )

#####################
# START MONTE CARLO #
#####################
tic = time.time()
for simiter in range(Nsim):

    # Report progress
    if simiter>0 and simiter%10==0:
        print("Iteration", simiter, "out of", Nsim)

    # Generate logistic regression data set
    X,y = GenerateLogisticDataTensorFlow(b, theta, n, rho=0.3)

    # Estimation
    bhat, thetahat, _, AsymptoticCovMatrix = EstimateLogisticRegressionNewtonRhapsonTensorFlow(X, y, bStart, thetaStart)

    # Store results
    ParaEst = ParaEst[:, simiter:(simiter+1)].assign( tf.concat([bhat, thetahat], 0) )
    tstatNum = tf.math.sqrt(tf.constant([[float(n)]]))*(tf.concat([bhat, thetahat], 0) - tf.concat([b, theta], 0) )
    tstatDen = tf.reshape(tf.math.sqrt(tf.linalg.diag_part(AsymptoticCovMatrix)), (p+1,1))
    tstat = tstat[:, simiter:(simiter+1)].assign( tstatNum/tstatDen )
toc = time.time()
print("Elapsed time is", toc-tic, "seconds.")

# Report figures
x = np.linspace(-5, 5, 100)
normpdf = norm.pdf(x, 0, 1)
fig, axs = plt.subplots(2, 1)
axs[0].hist(tstat[0,:].numpy(), bins=20, density=True, alpha=0.6, color='b')
axs[0].plot(x, normpdf, 'k', linewidth=2)
axs[1].hist(tstat[1,:].numpy(), bins=20, density=True, alpha=0.6, color='b')
axs[1].plot(x, normpdf, 'k', linewidth=2)
plt.show()
