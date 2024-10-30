import csv
import numpy as np
import os
import pandas as pd
import re
from scipy.stats import norm
from functions.LogisticRegressionFunctions import EstimateLogisticRegressionNewtonRhapsonNumPy, EstimateLogisticRegressionScikitLearn

# Specific function to print Haberman estimation to screen
def PrintResults2Screen(beta, theta, AsymptoticCovarianceMatrix, SampleSize, PrintHeader):
    print('\n\n'+PrintHeader)
    print('%-15s %-15s %-15s %-15s' % ('variable', 'MLE', 'Std. Error', 'P-value'))
    print('------------------------------------------------------------------')
    # Define variable names
    names = ['intercept', 'z_1', 'z_1^2', 'z_1^3', 'z_2', 'z_1xz_2', 'log(1+x_3)']
    # Stack parameters, compute standard erros and compute p-values
    gamma = np.hstack((beta, theta))
    StdError = np.sqrt(np.diag(AsymptoticCovarianceMatrix))/np.sqrt(SampleSize)
    Pvalues = 2*norm.cdf(-np.abs(gamma)/StdError)
    for iter in range(len(gamma)):
        print('%-15s %-15.4e %-15.4e %-15.4f' % (names[iter], gamma[iter], StdError[iter], Pvalues[iter]))

# Read input data
input_data = pd.read_csv("./HabermanDataset/haberman.txt", header=None, names=["Age", "Year", "AxilNodes", "y"])
input_data["y"] = -input_data["y"]+2

# Calculate standardized columns
input_data["z1"] = input_data["Age"]-52
input_data["z1^2"] = input_data["z1"]**2
input_data["z1^3"] = input_data["z1"]**3
input_data["z2"] = input_data["Year"]-63
input_data["z1_z2"] = input_data["z1"]*input_data["z2"]
input_data["log(1+AxilNodes)"] = np.log(1+input_data["AxilNodes"])

# Create X and y
X = input_data[["z1", "z1^2", "z1^3", "z2", "z1_z2", "log(1+AxilNodes)"]]
y = input_data["y"]
n = len(y)

# Remove observation 8 (see Landwehr, Pregibon and Shoemaker (1984), page 86)
X = X.drop(7)
y = y.drop(7)

# Estimation using NEWTON-RHAPSON in Numpy
bStart = 1.5; thetaStart = np.array([[0.03], [0.004], [-0.0003], [0], [0], [-0.5]]);
bhat1, thetahat1, _, AsymptoticCov1 = EstimateLogisticRegressionNewtonRhapsonNumPy(np.array(X).T, np.array(y).T, bStart, thetaStart)
PrintResults2Screen(bhat1, thetahat1.reshape(1,len(thetahat1))[0], AsymptoticCov1, n, 'NEWTON-RHAPSON in Numpy:')

# Estimation using SCIKIT-LEARN
bhat2, thetahat2, AsymptoticCov2 = EstimateLogisticRegressionScikitLearn(np.array(X), np.array(y))
PrintResults2Screen(bhat2, thetahat2, AsymptoticCov2, n, 'SCIKIT-LEARN:')


