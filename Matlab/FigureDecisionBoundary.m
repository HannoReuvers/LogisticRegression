clear variables; clc; close all;
addpath("./functions")
rng(1)

% Parameters
n = 50;
b = 0;
theta = [-5; 5];

% Sigmoid function
SigmoidFunction = @(x) 1./(1+exp(-x));

% Generate data
[X, y] = GenerateLogisticData(b, theta, n);

%%%%%%%%%%%%%%%%%%%%
% SIGMOID FUNCTION %
%%%%%%%%%%%%%%%%%%%%
xlist = -6:0.05:6;
ylist = SigmoidFunction(xlist);
plot(xlist, ylist, 'LineWidth', 2)
xticks([-5 -2.5 0 2.5 5])
yticks([0 0.25 0.5 0.75 1])
set(gca, 'FontSize', 12)
xlabel('x','FontSize',20) 
ylabel('$\Lambda(x)$','FontSize',20, 'Interpreter', 'latex') 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LINEAR DECISION BOUNDARY %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fit logistic regression
[bMLE, thetaMLE] = EstimateLogisticRegression(X, y, 0, zeros(2,1));

% Predictions on grid
x1min = -3; x1max = 3; x2min = -3; x2max = 3;
incr = 0.05;
x1Range = x1min:incr:x1max;
x2Range = x2min:incr:x2max;
[xx1, xx2] = meshgrid(x1Range,x2Range);
XGrid = [xx1(:) xx2(:)];
PredValues = NaN(size(XGrid,1), 1);
for point = 1:size(XGrid, 1)
    coordinates = XGrid(point, :);
    z= bMLE + coordinates*thetaMLE;
    PredProb = SigmoidFunction(z);
    PredValues(point) = 1.0*(PredProb>=0.5);
end

figure(2)
ColorScheme = colororder;
hold on
% Create canvas
scatter(XGrid(PredValues==1, 1), XGrid(PredValues==1, 2), 'o', 'MarkerEdgeColor', ColorScheme(1,:), 'MarkerFaceColor', ColorScheme(1,:), 'MarkerEdgeAlpha', 0.1, 'MarkerFaceAlpha', 0.1)
scatter(XGrid(PredValues==0, 1), XGrid(PredValues==0, 2), 'o', 'MarkerEdgeColor', ColorScheme(2,:), 'MarkerFaceColor', ColorScheme(2,:), 'MarkerEdgeAlpha', 0.1, 'MarkerFaceAlpha', 0.1)
% Add labeled data
plot(X(1,y == 1), X(2,y == 1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ColorScheme(1,:))
plot(X(1,y == 0), X(2,y == 0), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ColorScheme(2,:))
hold off
box on
set(gca, 'FontSize', 12)
xlabel('$x_1$', 'Interpreter', 'latex','FontSize', 25) 
ylabel('$x_2$', 'Interpreter', 'latex','FontSize', 25) 
axis([x1min x1max x2min x2max])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NONLINEAR DECISION BOUNDARY %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add square and cube of regressors
XExt = [X; X(1, :).^2; X(1, :).^3; X(2, :).^2; X(2, :).^3];

% Fit (extended) logistic regression
[bMLEExt, thetaMLEExt] = EstimateLogisticRegression(XExt, y, 0, zeros(size(XExt, 1),1));

% Predictions on grid
x1min = -3; x1max = 3; x2min = -3; x2max = 3;
incr = 0.05;
x1Range = x1min:incr:x1max;
x2Range = x2min:incr:x2max;
[xx1, xx2] = meshgrid(x1Range,x2Range);
XGrid = [xx1(:) xx2(:)];
PredValuesExt = NaN(size(XGrid,1), 1);
for point = 1:size(XGrid, 1)
    coordinates = XGrid(point, :);
    z= bMLEExt + [coordinates(1), coordinates(2), coordinates(1)^2, coordinates(1)^3, coordinates(2)^2, coordinates(2)^3]*thetaMLEExt;
    PredProb = SigmoidFunction(z);
    PredValuesExt(point) = 1.0*(PredProb>=0.5);
end

figure(3)
ColorScheme = colororder;
hold on
% Create canvas
scatter(XGrid(PredValuesExt==1, 1), XGrid(PredValuesExt==1, 2), 'o', 'MarkerEdgeColor', ColorScheme(1,:), 'MarkerFaceColor', ColorScheme(1,:), 'MarkerEdgeAlpha', 0.1, 'MarkerFaceAlpha', 0.1)
scatter(XGrid(PredValuesExt==0, 1), XGrid(PredValuesExt==0, 2), 'o', 'MarkerEdgeColor', ColorScheme(2,:), 'MarkerFaceColor', ColorScheme(2,:), 'MarkerEdgeAlpha', 0.1, 'MarkerFaceAlpha', 0.1)
% Add labeled data
plot(X(1,y == 1), X(2,y == 1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ColorScheme(1,:))
plot(X(1,y == 0), X(2,y == 0), 'o', 'MarkerSize', 10, 'MarkerFaceColor', ColorScheme(2,:))
hold off
box on
set(gca, 'FontSize', 12)
xlabel('$x_1$', 'Interpreter', 'latex','FontSize', 25) 
ylabel('$x_2$', 'Interpreter', 'latex','FontSize', 25) 
axis([x1min x1max x2min x2max])




