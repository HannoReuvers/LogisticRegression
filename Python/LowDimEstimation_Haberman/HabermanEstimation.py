import csv
import numpy as np
import os
import re
import tensorflow as tf
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

# Load function library
CurrentDirectory = os.path.dirname(__file__)
ParentDirectory = re.findall('\S+/Python', CurrentDirectory)[0]
LibraryScriptLocation = os.path.join(ParentDirectory, 'functions', 'LogisticRegressionFunctions.py')
exec(open(LibraryScriptLocation).read())

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

# Read data
rawdata= []
CurrentDirectory = os.path.dirname(__file__)
ParentDirectory = re.findall('\S+/LogisticRegression', CurrentDirectory)[0]
DataLocation = os.path.join(ParentDirectory, 'HabermanDataset', 'haberman.txt')
with open(DataLocation, 'r') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC, delimiter = ',')
    for row in reader:
        rawdata.append(row[:])
data = np.array(rawdata)

# Process variables
Age = data[:, 0]
Year = data[:, 1]
AxilNodes = data[:, 2]
y = data[:, 3]
y = -y+2
Z1 = Age-52
Z2 = Year-63

# Regressor matrix (with shape (306,6) )
X = np.stack((Z1, Z1**2, Z1**3, Z2, Z1*Z2, np.log(1+AxilNodes)), axis=1)

# Remove observation 8 (see Landwehr, Pregibon and Shoemaker (1984), page 86)
X_del = np.delete(X, obj=7, axis=0)
y_del = np.delete(y, obj=7, axis=0)
n = len(y_del)

# Estimation using NEWTON-RHAPSON in Numpy
bStart = 1.5; thetaStart = np.array([[0.03], [0.004], [-0.0003], [0], [0], [-0.5]]);
bhat1, thetahat1, _, AsymptoticCov1 = EstimateLogisticRegressionNewtonRhapsonNumPy(X_del.T, y_del.T, bStart, thetaStart)
PrintResults2Screen(bhat1, thetahat1.reshape(1,len(thetahat1))[0], AsymptoticCov1, n, 'NEWTON-RHAPSON in Numpy:')

# Estimation using SCIKIT-LEARN
bhat2, thetahat2, AsymptoticCov2 = EstimateLogisticRegressionScikitLearn(X_del, y_del)
PrintResults2Screen(bhat2, thetahat2, AsymptoticCov2, n, 'SCIKIT-LEARN:')
