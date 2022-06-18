clear variables; clc; close all;
addpath("./functions")

% Parameters
plist = (5:5:55)';
Nsim = 1E3;
rng(1)


MSElist = NaN(length(plist), 2);
timelist = NaN(length(plist), 2);
iter = 1;
for p = plist'
    disp(iter)
    [MSE1, time1] = Single_p_Sim(p, 0.3);
    [MSE2, time2] = Single_p_Sim(p, 0.8);
    MSElist(iter, :) = [MSE1 MSE2];
    timelist(iter, :) = [time1 time2];
    iter = iter+1;
end

figure(1)
c = get(gca, 'colororder');
plot(plist, MSElist(:, 1), 'o-', 'LineWidth', 2, 'MarkerFaceColor', c(1,:))
hold on
plot(plist, MSElist(:, 2), 'o-', 'LineWidth', 2, 'MarkerFaceColor', c(2,:))
hold off
box on
grid on
xticks([5 15 25 35 45 55])
yticks([0 0.2 0.4 0.6])
set(gca, 'FontSize', 15)
axis([5 55 0 0.6])
xlabel('p', 'FontSize', 20)
ylabel('MSE of $\hat \theta_1$', 'FontSize', 24, 'Interpreter', 'latex')


figure(2)
plot(plist, timelist(:, 1), 'o-', 'LineWidth', 2, 'MarkerFaceColor', c(1,:))
hold on
plot(plist, timelist(:, 2), 'o-', 'LineWidth', 2, 'MarkerFaceColor', c(2,:))
hold off
box on
grid on
xticks([5 15 25 35 45 55])
yticks([0 2 4 6 8 10])
set(gca, 'FontSize', 15)
axis([5 55 0 10])
xlabel('p', 'FontSize', 20)
ylabel('Computational time (s)', 'FontSize', 20)



function [MSEFirstTheta, ElapsedTime] = Single_p_Sim(p, rho)
%% DESCRIPTION: Simulate logistic regression data
%---INPUT VARIABLE(S)---
%   (1) p: number of features
%   (2) rho: correlation between normally distributed features
%---OUTPUT VARIABLE(S)---
%   (1) MSEFirstTheta: mean squared error of \theta_1
%   (2) ElapsedTime: required time to complete Monte Carlo simulation for
%   specific p

    % Simulation parameters
    n = 200;
    b = 0.5;
    theta = [0.3; 0.7];
    Nsim = 1E3;
    bStart = b;             % Start optimize from true parameter value

    % Initialise matrices
    ParaEst = NaN(p+1, Nsim);

    % Increase theta length
    theta = [theta; zeros(p-2,1)];
    thetaStart = theta;     % Start optimize from true parameter value

    tic
    for simiter = 1:Nsim

        % Generate logistic regression data set
        [X, y] = GenerateLogisticData(b, theta, n, rho);

        % Estimation
        [bhat, thetahat] = EstimateLogisticRegression(X, y, b, theta, false);

        % Store results
        ParaEst(:, simiter) = [bhat; thetahat];
    end


    % MSE computation
    MSEfull =  mean( (ParaEst - [b; theta]*ones(1, Nsim)).^2, 2);
    MSEFirstTheta = MSEfull(2);

    ElapsedTime = toc;
end
    


