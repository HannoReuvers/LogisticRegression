rm(list=ls())     # Clean memory
graphics.off()    # Close graphs
cat("\014")       # Clear Console
library(assertthat)
library(MASS)
library(tictoc)

# Load function librarysep
ScriptLocation <- paste0(dirname(rstudioapi::getSourceEditorContext()$path),"/functions/LogisticRegressionFunctions.R")
source(ScriptLocation)

# Parameters
method <- 'GLM'
n <- 500
b <- 0.5
theta <- matrix(c(0.3, 0.7), nrow=2, ncol=1)
Nsim <- 1000
bStart <- b
thetaStart <- theta

# Number of features
p <- length(theta)

# Initialize matrices
ParaEst <- matrix(NA, nrow=(p+1), ncol=Nsim)
tstat <- matrix(NA, nrow=(p+1), ncol=Nsim)

#####################
# START MONTE CARLO #
#####################
tic()
for(simiter in 1:Nsim)
{
    # Report progress
    if (simiter%%100==0) cat("Iteration", simiter, "out of", Nsim,"\n")
  
    # Generate logistic regression data 
    GeneratedData <- GenerateLogisticData(b, theta, n)
    X <- GeneratedData[[1]]
    y <- GeneratedData[[2]]
    
    if (method == 'NewtonRhapson')
    {
        EstimationOutput <- EstimateLogisticRegressionNewtonRhapson(X, y, bStart, thetaStart, FALSE, ComputeAsymptCov=TRUE)
        ParaEst[, simiter] <- c(EstimationOutput[[1]], EstimationOutput[[2]])
        tstat[, simiter] <- sqrt(n)*(ParaEst[, simiter] - matrix(c(b, theta) , nrow=p+1, ncol=1))/sqrt( diag(EstimationOutput[[4]]) )
    }
    else if (method == 'GLM')
    {
        EstimationOutput <- EstimateLogisticRegressionGLM(X, y, ComputeAsymptCov=TRUE)
        ParaEst[, simiter] <- EstimationOutput[[1]]
        tstat[, simiter] <- sqrt(n)*(ParaEst[, simiter] - matrix(c(b, theta) , nrow=p+1, ncol=1))/sqrt( diag(EstimationOutput[[2]]) )
    }
}
toc()

# Report figures
xlist <- seq(-5, 5, length = 200) 
ylist <- dnorm(xlist, mean = 0, sd = 1)

hist(tstat[1,], breaks=20, probability = TRUE) 
lines(xlist, ylist, col = "black")

hist(tstat[2,], breaks=20, probability = TRUE)
lines(xlist, ylist, col = "black")







