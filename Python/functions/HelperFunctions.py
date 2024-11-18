import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf


def GenerateRegressionData(b, theta, n: int, rho: float =  0.3):
    """
    Simulate regression data

    :param float b: Bias parameter
    :param np.array theta: Parameter vector
    :param int n: Sample size
    :param float rho: An (optional) parameter specifying the correlation between the normally distributed features
    :return X: The (pxn) regressor matrix (each rows contains values for a specific feature)
    :return y: The (1xn) vector with outcomes of the dependent variable
    """

    # Number of features
    p = len(theta)

    # Generate normally distributed features
    mu = np.zeros(p)
    Sigma = rho*np.ones( (p,p) ) + np.diag( (1-rho)*np.ones(p), 0 )
    X = np.zeros( (p,n) )
    for piter in range(n):
        X[:,piter] = np.random.multivariate_normal(mu, Sigma, size=None)

    # Generate errors and dependent variable
    errors = np.random.normal(0, size=(1, n))
    y = b + theta.T@X + errors

    return X, y


def GenerateLogisticData(b, theta, n, rho=0.3):
    """
    Simulate logistic regression data

    :param float b: Bias parameter
    :param np.array theta: Parameter vector
    :param int n: Sample size
    :param float rho: An (optional) parameter specifying the correlation between the normally distributed features
    :return X: The (pxn) regressor matrix (each rows contains values for a specific feature)
    :return y: The (1xn) vector with outcomes of the dependent variable
    """

    # Number of features
    p = len(theta)

    # Generate normally distributed features
    mu = np.zeros(p)
    Sigma = rho*np.ones( (p,p) ) + np.diag( (1-rho)*np.ones(p), 0 )
    X = np.zeros( (p,n) )
    for piter in range(n):
        X[:,piter] = np.random.multivariate_normal(mu, Sigma, size=None)

    #--- Generate categorical outcome ---#
    y = 1.0*( np.random.rand(1,n) <= 1/(1+np.exp(-(b+theta.T@X))) )

    # Return output
    return X, y


def GenerateLogisticDataTensorFlow(b, theta, n, rho=0.3):
    """
    Simulate logistic regression data using TensorFlow

    :param float b: Bias parameter
    :param np.array theta: Parameter vector
    :param int n: Sample size
    :param float rho: An (optional) parameter specifying the correlation between the normally distributed features
    :return X: The (pxn) regressor matrix (each rows contains values for a specific feature)
    :return y: The (1xn) vector with outcomes of the dependent variable
    """

    # Number of features
    p = len(theta)

    #--- Generate normally distributed features ---#
    Sigma = rho*tf.ones( (p,p) ) + tf.linalg.diag( (1-rho)*tf.ones(p), 0 )
    CholSigma = tf.linalg.cholesky(Sigma)
    X = tf.Variable( tf.zeros((p,n)) )
    # Generate (pxn) regressor matrix
    for piter in range(n):
        X = X[:, piter:(piter+1)].assign(tf.linalg.matmul(CholSigma, tf.random.normal((p,1))))

    #--- Generate categorical outcome ---#
    Prob = 1/( 1+tf.math.exp(-(b+tf.linalg.matmul(theta, X, transpose_a=True))))
    y = tf.Variable( tf.zeros((1,n)) )
    for iter in range(n):
        if tf.math.greater(Prob[0, iter], tf.random.uniform((1,1))[0,0]):
            y = y[0, iter].assign(1)

    # Return output
    return X, y


def EstimateLogisticRegressionNewtonRhapsonNumPy(X, y, bStart, thetaStart, DisplayIterationDetails=False, ComputeAsymptCov=True):
    """
    Estimate logistic regression model using Newton-Rhapson in Numpy

    :param np.array X: The (pxn) regressor matrix (each rows contains values for a specific feature)
    :param np.array y: The (1xn) vector with outcomes of the dependent variable
    :param np.array bStart: The starting guess for the bias parameter
    :param np.array thetaStart: The starting guess for theta
    :param boolean DisplayIterationDetails: Optional parameter to print convergence details
    :param boolean ComputeAsymptCov: Optional parameter to compute the asymptotic covariance matrix
    :return bhat: MLE for bias
    :return thetahat: MLE for theta
    :return ConvergenceAchieved: Boolean output reporting whether or not Newton-Rhapson converged
    :return AsymptCov: A consistent estimator of asymptotic covariance matrix of MLE (NaN if ComputeAsymptCov=False)
    """

    # Newton-Rhapson algorithm parameters
    MAXITER = 5000
    RELTOL = 1E-6

    # Dimensions
    p, n = X.shape

    #--- NEWTON-RHAPSON ALGORITHM ---#
    bOLD = bStart
    thetaOLD = thetaStart
    ConvergenceAchieved = False
    if DisplayIterationDetails:
        print('\n%-7s %-15s %-15s %-15s' % ("iter", "log-L", "rel. delta b", "rel. delta theta"))
        print('------------------------------------------------------------------')
    # Iterations
    for iter in range(MAXITER):

        # Evaluate probabilities
        LambdaProb = 1/( 1+np.exp(-(2*y-1)*(bOLD+thetaOLD.T@X)) )

        # Gradient calculation
        Gradb = np.inner(2*y-1, 1-LambdaProb)
        Gradtheta = X@( (2*y-1)*(1-LambdaProb) ).T
        Grad = np.vstack((Gradb,Gradtheta))/n

        # Hessian calculation
        Hessian = np.zeros((p+1,p+1))
        for niter in range(n):
            z = np.vstack( (np.ones(1),X[:,niter:(niter+1)]))
            Hessian = Hessian - np.outer(z,z)*LambdaProb[0,niter]*(1-LambdaProb[0,niter])
        Hessian = Hessian/n

        # Newton-Rhapson update
        NewPara = np.vstack( (bOLD,thetaOLD) ) - np.linalg.solve(Hessian, Grad)
        bNEW = NewPara[0,0]
        thetaNEW = NewPara[-p:]

        # Relative parameter change
        bRelChange = np.absolute(bNEW-bOLD)/np.absolute(bOLD)
        thetaRelChange = np.linalg.norm(thetaNEW-thetaOLD)/np.linalg.norm(thetaOLD)

        # Report convergence metrics (OPTIONAl)
        if DisplayIterationDetails:
            print("%-7d %-15.4f %-15.4g %-15.4g" % (iter, np.sum(np.log(LambdaProb)), bRelChange, thetaRelChange))

        # Check convergence
        if (bRelChange<RELTOL) and (thetaRelChange<RELTOL):
            ConvergenceAchieved = True
            bhat = bNEW
            thetahat = thetaNEW
            break

        # Update parameters for next iteration
        bOLD = bNEW
        thetaOLD = thetaNEW

    # Convergence notification (OPTIONAL)
    if DisplayIterationDetails:
        print('\n\nConvergence after', iter, 'out of', MAXITER, 'iterations')

    # Check convergence
    assert ConvergenceAchieved, f"Newton-Rhapson algorithm did not converge. Increase MAXITER and/or check for other problems..."

    #--- ASYMPTOTIC COVARIANCE MATRIX (OPTIONAL) ---#
    if ComputeAsymptCov:
        hatLambdaProb = 1/( 1+np.exp(-(2*y-1)*(bhat+thetahat.T@X)) )
        hatHessian = np.zeros((p+1,p+1))
        for niter in range(n):
            z = np.vstack( (np.ones(1),X[:,niter:(niter+1)]))
            hatHessian = hatHessian - np.outer(z,z)*hatLambdaProb[0,niter]*(1-hatLambdaProb[0,niter])
        hatHessian = hatHessian/n
        AsymptCov = -np.linalg.inv(hatHessian)
    else:
        AsymptCov = np.NaN

    return bhat, thetahat, ConvergenceAchieved, AsymptCov


def EstimateLogisticRegressionNewtonRhapsonTensorFlow(X, y, bStart, thetaStart, DisplayIterationDetails=False, ComputeAsymptCov=True):
    """
    Estimate logistic regression model using Newton-Rhapson in TensorFlow

    :param np.array X: The (pxn) regressor matrix (each rows contains values for a specific feature)
    :param np.array y: The (1xn) vector with outcomes of the dependent variable
    :param np.array bStart: The starting guess for the bias parameter
    :param np.array thetaStart: The starting guess for theta
    :param boolean DisplayIterationDetails: Optional parameter to print convergence details
    :param boolean ComputeAsymptCov: Optional parameter to compute the asymptotic covariance matrix
    :return bhat: MLE for bias
    :return thetahat: MLE for theta
    :return ConvergenceAchieved: Boolean output reporting whether or not Newton-Rhapson converged
    :return AsymptCov: A consistent estimator of asymptotic covariance matrix of MLE (NaN if ComputeAsymptCov=False)
    """

    # Newton-Rhapson algorithm parameters
    MAXITER = 5000
    RELTOL = 1E-6

    # Dimensions
    p, n = X.shape

    #--- NEWTON-RHAPSON ALGORITHM ---#
    bOLD = bStart
    thetaOLD = thetaStart
    ConvergenceAchieved = False
    if DisplayIterationDetails:
        print('\n%-7s %-15s %-15s %-15s' % ("iter", "log-L", "rel. delta b", "rel. delta theta"))
        print('------------------------------------------------------------------')
    # Iterations
    for iter in range(MAXITER):

        # Evaluate probabilities
        LambdaProb = 1/( 1+tf.math.exp(-(2*y-1)*(bOLD+tf.linalg.matmul(thetaOLD, X, transpose_a=True))) )

        # Gradient calculation
        Gradb = tf.linalg.matmul(2*y-1, 1-LambdaProb, transpose_b=True)
        Gradtheta = tf.linalg.matmul(X, (2*y-1)*(1-LambdaProb), transpose_b=True)
        Grad = tf.concat([Gradb, Gradtheta], 0)/n

        # Hessian calculation
        Hessian = tf.zeros([p+1,p+1])
        for niter in range(n):
            z = tf.concat([tf.ones([1, 1]), X[:, niter:(niter+1)]], 0)
            Hessian = Hessian - tf.linalg.matmul(z,z, transpose_b=True)*LambdaProb[0,niter]*(1-LambdaProb[0,niter])
        Hessian = Hessian/n

        # Newton-Rhapson update
        NewPara = tf.concat([bOLD,thetaOLD], 0) - tf.linalg.solve(Hessian, Grad)
        bNEW = NewPara[0:1,0:1]
        thetaNEW = NewPara[-p:]

        # Relative parameter change
        bRelChange = tf.norm(bNEW-bOLD)/tf.norm(bOLD)
        thetaRelChange = tf.norm(thetaNEW-thetaOLD)/tf.norm(thetaOLD)

        # Report convergence metrics (OPTIONAl)
        if DisplayIterationDetails:
            print("%-7d %-15.4f %-15.4g %-15.4g" % (iter, tf.math.reduce_sum(tf.math.log(LambdaProb), axis=0)[0], bRelChange, thetaRelChange))

        # Check convergence
        if (bRelChange<RELTOL) and (thetaRelChange<RELTOL):
            ConvergenceAchieved = True
            bhat = bNEW
            thetahat = thetaNEW
            break

        # Update parameters for next iteration
        bOLD = bNEW
        thetaOLD = thetaNEW

    # Convergence notification (OPTIONAL)
    if DisplayIterationDetails:
        print('\n\nConvergence after', iter, 'out of', MAXITER, 'iterations')

    # Check convergence
    assert ConvergenceAchieved, f"Newton-Rhapson algorithm did not converge. Increase MAXITER and/or check for other problems..."

    #--- ASYMPTOTIC COVARIANCE MATRIX (OPTIONAL) ---#
    if ComputeAsymptCov:
        hatLambdaProb = 1/( 1+tf.math.exp(-(2*y-1)*(bhat+tf.linalg.matmul(thetahat, X, transpose_a=True))) )
        hatHessian = tf.zeros([p+1,p+1])
        for niter in range(n):
            z = tf.concat([tf.ones([1, 1]), X[:, niter:(niter+1)]], 0)
            hatHessian = hatHessian - np.outer(z,z)*hatLambdaProb[0,niter]*(1-hatLambdaProb[0,niter])
        hatHessian = hatHessian/n
        AsymptCov = -tf.linalg.inv(hatHessian)
    else:
        AsymptCov = tf.constant([np.nan])

    return bhat, thetahat, ConvergenceAchieved, AsymptCov


def EstimateLogisticRegressionScikitLearn(X, y, ComputeAsymptCov=True):
    """
    Estimate logistic regression model using Scikit-learn

    :param np.array X: The (pxn) regressor matrix (each rows contains values for a specific feature)
    :param np.array y: The (1xn) vector with outcomes of the dependent variable
    :param boolean ComputeAsymptCov: Optional parameter to compute the asymptotic covariance matrix
    :return bhat: MLE for bias
    :return thetahat: MLE for theta
    :return AsymptCov: A consistent estimator of asymptotic covariance matrix of MLE (NaN if ComputeAsymptCov=False)
    """

    # Dimensions
    n = X.shape[0]
    p = X.shape[1]

    # Algorithm parameters
    MAXITER = 5000
    TOL = 1E-6

    # Estimate logistic regression (include: from sklearn.linear_model import LogisticRegression)
    clf = LogisticRegression(random_state=0, penalty=None, max_iter=MAXITER, tol = TOL, solver = 'lbfgs').fit(X, y)

    # Select parameters
    bhat = clf.intercept_
    thetahat = clf.coef_[0]

    # Covariance matrix
    if ComputeAsymptCov:
        hatHessian = np.zeros((p+1, p+1))
        hatLambdaProb = clf.predict_proba(X)[:,0]
        for niter in range(n):
            z = np.reshape( np.hstack((1, X[niter,:].T)), newshape=(p+1,1) )
            hatHessian = hatHessian - tf.linalg.matmul(z,z, transpose_b=True)*hatLambdaProb[niter]*(1-hatLambdaProb[niter])
        hatHessian = hatHessian/n
        AsymptCov = - np.linalg.inv(hatHessian)
    else:
        AsymptCov = np.NaN

    # Return output
    return bhat, thetahat, AsymptCov


def soft_thresholding(x, threshold):
    if np.abs(x)<threshold:
        return 0
    else:
        return np.sign(x)*(np.abs(x)-threshold)


def LinearRegressionLasso(X, y, lambdaParameter, bStart, thetaStart, DisplayIterationDetails=False):
    """
    Estimate a linear regression model with an L1 penalty on the coefficients (intercept is not penalized).
    """

    print(DisplayIterationDetails)

    # Dimensions
    p, n = X.shape

    # Algorithm parameters
    MAXITER = 5000
    RELTOL = 1E-6

    # Calculate average of squared regressor value outside of main loop
    avg_squares_X = np.mean(X**2, axis=1, keepdims=True)

    #--- COORDINATE-DESCENT ---#
    bOLD = bStart
    thetaOLD = thetaStart
    thetaNEW = np.copy(thetaOLD)
    ConvergenceAchieved = False
    if DisplayIterationDetails:
        print('\n%-7s %-15s %-15s %-15s' % ("iter", "obj. func.", "rel. delta b", "rel. delta theta"))
        print('------------------------------------------------------------------')
    # Iterations
    for iter in range(MAXITER):

        # Update intercept estimate
        bNEW = np.mean(y - thetaOLD.T@X)
    
        # Coordinate descent for feature parameters
        for k in range(p):

            # Compute r_{-k}
            thetaTEMP = np.copy(thetaNEW)
            thetaTEMP[k] = 0
            r_min_k = y - bNEW - thetaTEMP@X

            # Apply coordinate descent formula to element k of theta
            thetaNEW[k] = (soft_thresholding(r_min_k@X[k,:]/n, lambdaParameter)/avg_squares_X[k])[0]

        # Relative parameter change (the case thetaNEW=thetaOLD=np.zeros should be treated explicitly)
        bRelChange = np.absolute(bNEW-bOLD)/np.absolute(bOLD)
        if (np.linalg.norm(thetaOLD)<1E-10) and (np.linalg.norm(thetaNEW-thetaOLD)<1E-10):
            thetaRelChange = 0
        else:
            thetaRelChange = np.linalg.norm(thetaNEW-thetaOLD)/np.linalg.norm(thetaOLD)


        # Report convergence metrics (OPTIONAl)
        if DisplayIterationDetails:
            squaredErrors = (y - bNEW- thetaNEW.T@X)@(y - bNEW- thetaNEW.T@X).T/(2*n)
            parL1Norm = sum(abs(thetaNEW))
            #print(squaredErrors)
            #print(parL1Norm)
            #print(thetaNEW)
            print("%-7d %-15.4f %-15.4g %-15.4g" % (iter, squaredErrors+lambdaParameter*parL1Norm, bRelChange, thetaRelChange))

        # Check convergence
        if (bRelChange<RELTOL) and (thetaRelChange<RELTOL):
            ConvergenceAchieved = True
            bhat = bNEW
            thetahat = thetaNEW
            break

        # Update parameters for next iteration
        bOLD = np.copy(bNEW)
        thetaOLD = np.copy(thetaNEW)

    return bhat, thetahat, ConvergenceAchieved

