function [X, y] = GenerateLogisticData(b, theta, n, rho)
%% DESCRIPTION: Simulate logistic regression data
%---INPUT VARIABLE(S)---
%   (1) b: bias parameter
%   (2) theta: parameter vector
%   (3) n: sample size
%   (4) rho: correlation parameter for regressions (default value is 0.3)
%---OUTPUT VARIABLE(S)---
%   (1) X: (pxn) matrix with features in columns
%   (2) y: (1xn) data series with 0 or 1 outcome

    % Default value for feature correlation
    if nargin<4
        rho = 0.3;
    end

    % Number of features
    p = length(theta);

    %--- Generate normally distributed features ---%
    mu = zeros(p ,1);
    Sigma = rho*ones(p, p);
    Sigma = Sigma + diag(ones(p, 1) -diag(Sigma));
    X = zeros(p, n);
    % Generate (pxn) regressor matrix
    for piter = 1:n
        X(:, piter) = mvnrnd(mu, Sigma);
    end

    %--- Generate categorical outcome ---%
    y = 1.0*( rand(1,n) <= 1./(1+exp(-(b+theta'*X))) );
end

