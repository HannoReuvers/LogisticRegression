################################### FUNCTION: GenerateLogisticData ###################################
def GenerateLogisticData(b, theta, n, rho=0.3):
    """
    DESCRIPTION: Simulate logistic regression data
    --- INPUT VARIABLE(S) ---
    (1) b: bias parameter
    (2) theta: parameter vector
    (3) n: sample size
    (4) rho (OPTIONAL): correlation parameter for regressions
    --- OUTPUT VARIABLE(S) ---
    (1) X: matrix with features in columns ( shape = (p,n) )
    (2) y: data series with 0 or 1 outcome ( shape = (1,n) )
    """

    # Number of features
    p = len(theta)

    #--- Generate normally distributed features ---#
    mu = np.zeros(p)
    Sigma = rho*np.ones( (p,p) ) + np.diag( (1-rho)*np.ones(p), 0 )
    X = np.zeros( (p,n) )
    # Generate (pxn) regressor matrix
    for piter in range(n):
        X[:,piter] = np.random.multivariate_normal(mu, Sigma, size=None)

    #--- Generate categorical outcome ---#
    y = 1.0*( np.random.rand(1,n) <= 1/(1+np.exp(-(b+theta.T@X))) )

    # Return output
    return X, y

################################### FUNCTION: GenerateLogisticDataTensorFlow ###################################
def GenerateLogisticDataTensorFlow(b, theta, n, rho=0.3):
    """
    DESCRIPTION: Simulate logistic regression data using TensorFlow
    --- INPUT VARIABLE(S) ---
    (1) b: bias parameter
    (2) theta: parameter vector
    (3) n: sample size
    (4) rho (OPTIONAL): correlation parameter for regressions
    --- OUTPUT VARIABLE(S) ---
    (1) X: tensor with features in columns ( shape = (p,n) )
    (2) y: tensor data series with 0 or 1 outcome ( shape = (1,n) )
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

################################### FUNCTION: EstimateLogisticRegressionNewtonRhapsonNumPy ###################################
def EstimateLogisticRegressionNewtonRhapsonNumPy(X, y, bStart, thetaStart, DisplayIterationDetails=False, ComputeAsymptCov=True):
    """
    DESCRIPTION: Estimate logistic regression model using Newton-Rhapson in Numpy
    --- INPUT VARIABLE(S) ---
    (1) X: matrix with features in columns ( shape = (p,n) )
    (2) y: data series with 0 or 1 outcome ( shape = (1,n) )
    (3) bStart: starting guess for bias
    (4) thetaStart: starting guess for theta
    (5) DisplayIterationDetails (OPTIONAL): display convergence details?
    (6) ComputeAsymptCov: compute asymptotic covariance matrix?
    --- OUTPUT VARIABLE(S) ---
    (1) bhat: MLE for bias
    (2) thetahat: MLE for theta
    (3) ConvergenceAchieved: did Newton-Rhapson converge?
    (4) AsymptCov: consistent estimator of asymptotic covariance matrix of MLE (NaN if ComputeAsymptCov=False)
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

################################### FUNCTION: EstimateLogisticRegressionNewtonRhapsonTensorFlow ###################################
def EstimateLogisticRegressionNewtonRhapsonTensorFlow(X, y, bStart, thetaStart, DisplayIterationDetails=False, ComputeAsymptCov=True):
    """
    DESCRIPTION: Estimate logistic regression model using Newton-Rhapson in TensorFlow
    --- INPUT VARIABLE(S) ---
    (1) X: matrix with features in columns ( shape = (p,n) )
    (2) y: data series with 0 or 1 outcome ( shape = (1,n) )
    (3) bStart: starting guess for bias
    (4) thetaStart: starting guess for theta
    (5) DisplayIterationDetails (OPTIONAL): display convergence details?
    (6) ComputeAsymptCov: compute asymptotic covariance matrix?
    --- OUTPUT VARIABLE(S) ---
    (1) bhat: MLE for bias
    (2) thetahat: MLE for theta
    (3) ConvergenceAchieved: did Newton-Rhapson converge?
    (4) AsymptCov: consistent estimator of asymptotic covariance matrix of MLE (NaN if ComputeAsymptCov=False)
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

################################### FUNCTION: EstimateLogisticRegressionScikitLearn ###################################
def EstimateLogisticRegressionScikitLearn(X, y, ComputeAsymptCov=True):
    """
    DESCRIPTION: Estimate logistic regression model using Scikit-learn
    --- INPUT VARIABLE(S) ---
    (1) X: (nxp) matrix with features in rows
    (2) y: (nx1) data series with 0 or 1 outcome
    (3) ComputeAsymptCov: compute asymptotic covariance matrix?
    --- OUTPUT VARIABLE(S) ---
    (1) bhat: MLE for bias
    (2) thetahat: MLE for theta
    (3) AsymptCov: consistent estimator of the asymptotic covariance matrix of the MLE
    """

    # Dimensions
    n = X.shape[0]
    p = X.shape[1]

    # Algorithm parameters
    MAXITER = 5000
    TOL = 1E-6

    # Estimate logistic regression (include: from sklearn.linear_model import LogisticRegression)
    clf = LogisticRegression(random_state=0, penalty='none', max_iter=MAXITER, tol = TOL, solver = 'lbfgs').fit(X, y)

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
