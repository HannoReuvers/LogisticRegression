function [bhat, thetahat, ConvergenceAchieved, AsymptCov] = EstimateLogisticRegression(X, y, bStart, thetaStart, DisplayIterationDetails)
%% DESCRIPTION: Estimate Logistic Regression Model
%---INPUT VARIABLE(S)---
%   (1) X: (pxn) matrix with features in columns
%   (2) y: (1xn) data series with 0 or 1 outcome
%   (3) bStart: starting guess for bias
%   (4) thetaStart: starting guess for theta vector
%   (5) DisplayIterationDetails: display convergence details?
%---OUTPUT VARIABLE(S)---
%   (1) bhat: MLE for bias
%   (2) thetahat: MLE for theta
%   (3) ConvergenceAchieved: did Newton-Rhapson converge?
%   (4) AsymptCov: consistent estimator of asymptotic covariance matrix
%   of MLE

    % Default value of iteration display
    if nargin<5
        DisplayIterationDetails = false;
    end

    % Newton-Raphson algorithm parameters
    MAXITER = 5000;
    RELTOL = 1E-6;

    % Dimensions
    [p, n] = size(X);

    %--- NEWTON-RHAPSON ALGORITHM ---%
    bOLD = bStart;
    thetaOLD = thetaStart;
    ConvergenceAchieved = false;
    if DisplayIterationDetails==true
        fprintf("%-7s %-15s %-15s %-15s\n", "iter", "log-L", "rel. delta b", "rel. delta theta")
        fprintf('------------------------------------------------------------------\n')
    end
    % Iterations
    for iter = 1:MAXITER

        % Evaluate probabilities
        LambdaProb = 1./( 1+exp(-(2*y-1).*(bOLD+thetaOLD'*X)) );

        % Gradient calculation
        Gradb = (2*y-1)*(1-LambdaProb)';
        Gradtheta = X*((2*y-1).*(1-LambdaProb))';
        Grad = [Gradb; Gradtheta]/n;

        % Hessian calculation
        Hessian = zeros(p+1);
        for niter = 1:n
            z = [1; X(:,niter)];
            Hessian = Hessian - (z*z')*LambdaProb(niter)*(1-LambdaProb(niter));
        end
        Hessian = Hessian/n;

        % Newton-Rhapson update
        NewPara = [bOLD; thetaOLD] - Hessian\Grad;
        bNEW = NewPara(1);
        thetaNEW = NewPara(2:end);

        % Relative parameter changes
        bRelChange = abs(bNEW-bOLD)/abs(bOLD);
        thetaRelChange = norm(thetaNEW-thetaOLD)/norm(thetaOLD);

        % Report convergence metrics (OPTIONAL)
        if DisplayIterationDetails==true
            fprintf("%-7d %-15.4f %-15.4g %-15.4g\n", iter, sum(log(LambdaProb)), bRelChange, thetaRelChange)
        end

        % Check convergence
        if (bRelChange< RELTOL && thetaRelChange < RELTOL )
            ConvergenceAchieved = true;
            bhat = bNEW;
            thetahat = thetaNEW;
            break
        end

        % Update parameters for next iteration
        bOLD = bNEW;
        thetaOLD = thetaNEW;
    end

    % Convergence notification (OPTIONAL)
    if DisplayIterationDetails==true
        fprintf("\n\nConvergence after %d out of %d iterations\n\n\n", iter, MAXITER)
    end
    
    % Check convergence
    assert(ConvergenceAchieved==true, "Newton-Rhapson algorithm did not converge. Increase MAXITER and/or check for other problems...")

    %--- ASYMPTOTIC COVARIANCE MATRIX (OPTIONAL) ---%
    if nargout>3
        hatLambdaProb = 1./( 1+exp(-(2*y-1).*(bhat+thetahat'*X)) );
        hatHessian = zeros(p+1);
        for niter = 1:length(y)
            z = [1; X(:,niter)];
            hatHessian = hatHessian - (z*z')*hatLambdaProb(niter)*(1-hatLambdaProb(niter));
        end
        hatHessian = hatHessian/n;
        AsymptCov = - inv(hatHessian);
    end
end

