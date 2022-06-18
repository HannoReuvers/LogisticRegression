clear variables; clc; close all;
addpath("./functions")

% Parameters
n = 500;
b = 0.5;
theta = [0.3; 0.7];
Nsim = 1E3;
rng(1)
bStart = b;             % Start optimize from true parameter value
thetaStart = theta;     % Start optimize from true parameter value

% Number of features
p = length(theta);

% Initialise matrices
ParaEst = NaN(p+1, Nsim);
tstat = NaN(p+1, Nsim);

%%%%%%%%%%%%%%%%%%%%%
% START MONTE CARLO %
%%%%%%%%%%%%%%%%%%%%%
tic
for simiter = 1:Nsim

    % Report progress
    if (mod(simiter,1E2)) == 0
        fprintf('Iteration %5d out of %5d \n', simiter, Nsim);
    end

    % Generate logistic regression data set
    [X, y] = GenerateLogisticData(b, theta, n);

    % Estimation
    [bhat, thetahat, ~, AsymptCovMatrix] = EstimateLogisticRegression(X, y, b, theta, false);
    % Store results
    ParaEst(:, simiter) = [bhat; thetahat];
    tstat(:, simiter) = sqrt(n)*( ParaEst(:, simiter)-[b; theta] )./sqrt( diag(AsymptCovMatrix) );
end
toc

% Report figures
figure(1)
subplot(2,1,1)
histogram(tstat(1, :), 20, 'Normalization', 'pdf') % Ignores NaN
hold on
xlist = -5:0.05:5;
ylist = normpdf(xlist);
plot(xlist, ylist, 'LineWidth', 2)
hold off
axis([-5 5 0 0.5])
subplot(2,1,2)
histogram(tstat(2, :), 20, 'Normalization', 'pdf') % Ignores NaN
hold on
xlist = -5:0.05:5;
ylist = normpdf(xlist);
plot(xlist, ylist, 'LineWidth', 2)
hold off
axis([-5 5 0 0.5])
    


