################################### FUNCTION: GenerateLogisticData ###################################
#---INPUT VARIABLE(S)---
#   (1) b: bias parameter
#   (2) theta: parameter vector
#   (3) n: sample size
#   (4) rho (OPTIONAL): correlation parameter for regressors
#---OUTPUT VARIABLE(S)---
#   (1) X: matrix with features in columns (nrow=p, ncol=n)
#   (2) y: vector with 0 or 1 outcome (nrow=1, ncol=n)
GenerateLogisticData <- function(b, theta, n, rho=0.3)
{
    # Number of features
    p <- length(theta)
    
    #--- Generate normally distributed features ---#
    Sigma = matrix(data = rho, nrow = p, ncol = p)
    diag(Sigma) <- 1
    # Generate (pxn) regressor matrix
    X <- t(mvrnorm( n, matrix(0, nrow=1, ncol=p), Sigma ))
    
    # Generate categorical oucome
    y <- 1.0*( runif(n, min = 0, max = 1) <= 1/(1 + exp(- (b+crossprod(theta, X) ) ) )  )
    
    # Return output
    return(list(X,y))
}

################################### FUNCTION: EstimateLogisticRegressionNewtonRhapson ###################################
#---INPUT VARIABLE(S)---
#   (1) X: matrix with features in columns (nrow=p, ncol=n)
#   (2) y: vector with 0 or 1 outcome (nrow=1, ncol=n)
#   (3) ComputeAsymptCov: compute asymptotic covariance matrix?
#---OUTPUT VARIABLE(S)---
#   (1) X: matrix with features in columns (nrow=p, ncol=n)
#   (2) y: vector with 0 or 1 outcome (nrow=1, ncol=n)
EstimateLogisticRegressionNewtonRhapson <- function(X, y, bStart, thetaStart, DisplayIterationDetails=FALSE, ComputeAsymptCov=TRUE)
{
    # Newton-Rhapson algorithm parameters
    MAXITER = 5000
    RELTOL = 1E-5
    
    # Dimensions
    p = dim(X)[1]
    n = dim(X)[2]
    
    #--- NEWTON-RHAPSON ALGORITHM ---#
    bOLD = bStart
    thetaOLD = thetaStart
    ConvergenceAchieved = FALSE
    if (DisplayIterationDetails==TRUE)
    {
        cat(sprintf('\n%-7s %-15s %-15s %-15s', "iter", "log-L", "rel. delta b", "rel. delta theta"))
        sprintf('------------------------------------------------------------------')
    }
    # Iterations
    for (iter in 1:MAXITER)
    {
        # Evaluate probabilities
        LambdaProb <- matrix(1/( 1+exp( -(2*y-1)*(bOLD+t(thetaOLD)%*%X)) ), nrow=1, ncol=n)
        
        # Gradient calculation
        Gradb <- tcrossprod(2*y-1, 1-LambdaProb)
        Gradtheta <- tcrossprod(X, (2*y-1)*(1-LambdaProb) )
        Grad <- matrix(c(Gradb, Gradtheta), nrow=p+1, ncol=1)
        Grad <- Grad/n
        
        
        # Hessian calculation
        Hessian <- matrix(0, nrow=p+1, ncol=p+1)
        for (niter in 1:n)
        {
            z <- matrix(c(1,X[,niter]), nrow=p+1, ncol=1)
            Hessian <- Hessian-tcrossprod(z)*LambdaProb[niter]*(1-LambdaProb[niter])
        }
        Hessian <- Hessian/n
        
        # Newton-Rhapson update
        NewPara <- matrix(c(bOLD, thetaOLD), nrow=p+1, ncol=1)-solve(Hessian, Grad)
        bNEW <- NewPara[1]
        thetaNEW <- NewPara[2:(p+1)]
        
        # Relative parameter change
        bRelChange <- abs(bNEW-bOLD)/abs(bOLD)
        thetaRelChange <- norm(thetaNEW-thetaOLD, type="2")/norm(thetaOLD, type="2")
        
        # Report convergence metrics (OPTIONAL)
        if (DisplayIterationDetails==TRUE)
        {
            cat(sprintf('\n%-7d %-15.4f %-15.4g %-15.4g', iter, sum(log(LambdaProb)), bRelChange, thetaRelChange))
        }
        
        # Check convergence
        if (bRelChange<RELTOL && thetaRelChange<RELTOL)
        {
            ConvergenceAchieved <- TRUE
            bhat <- bNEW
            thetahat <- thetaNEW
            break
        }
        
        # Update parameters for next iteration
        bOLD <- bNEW
        thetaOLD <- thetaNEW
    }
    
    # Convergence notification (OPTIONAL)
    if (DisplayIterationDetails==TRUE)
    {
        cat(sprintf('\n\nConvergence after %d out of %d iterations', iter, MAXITER))
    }
    
    # Check convergence
    stopifnot(ConvergenceAchieved==TRUE)
    
    #--- ASYMPTOTIC COVARIANCE MATRIX (OPTIONAL) ---#
    if (ComputeAsymptCov==TRUE)
    {
        hatLambdaProb <- matrix(1/( 1+exp( -(2*y-1)*(bhat+t(thetahat)%*%X)) ), nrow=1, ncol=n)
        hatHessian <- matrix(0, nrow=p+1, ncol=p+1)
        for (niter in 1:n)
        {
          z <- matrix(c(1,X[,niter]), nrow=p+1, ncol=1)
          hatHessian <- hatHessian-tcrossprod(z)*hatLambdaProb[niter]*(1-hatLambdaProb[niter])
        }
        hatHessian <- hatHessian/n
        AsymptCov <- -solve(hatHessian)
    }
    else
    {
        AsymptCov <- NA
    }

    # Return output
    return(list(bhat, thetahat, ConvergenceAchieved, AsymptCov))   
}

################################### FUNCTION: EstimateLogisticRegressionGLM ###################################
#---INPUT VARIABLE(S)---
#   (1) X: matrix with features in columns (nrow=p, ncol=n)
#   (2) y: vector with 0 or 1 outcome (nrow=1, ncol=n)
#   (3) ComputeAsymptCov: compute asymptotic covariance matrix?
#---OUTPUT VARIABLE(S)---
#   (1) X: matrix with features in columns (nrow=p, ncol=n)
#   (2) y: vector with 0 or 1 outcome (nrow=1, ncol=n)
EstimateLogisticRegressionGLM <- function(X, y, ComputeAsymptCov=TRUE)
{
    # Dimensions
    p = dim(X)[1]
    n = dim(X)[2]
  
    # Fit model
    model <- glm( t(y) ~., data = data.frame(t(X)), family = binomial)
    gammahat <- model$coefficients      # MLE of full parameter vector (length p+1)
    
    # Covariance matrix
    if (ComputeAsymptCov==TRUE)
    {
        hatHessian = matrix(0, ncol=p+1, nrow=p+1)
        hatLambdaProb <- model$fitted.values
        for (niter in 1:n)
        {
            hatHessian <- hatHessian - tcrossprod(matrix(c(1,X[,niter])))*hatLambdaProb[niter]*(1-hatLambdaProb[niter])
        }
        hatHessian <- hatHessian/n
        AsymptCov <- -solve(hatHessian)
    }

    # Return output
    return(list(gammahat,AsymptCov))
}
# Something
print("Hello")