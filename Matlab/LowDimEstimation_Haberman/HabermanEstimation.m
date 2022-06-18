clear variables; clc; close all;

% Load function directory
addpath("../functions")

% Read data from SQLite
sqlfile = fullfile('../../HabermanDataset/', 'HabermanDataSet.sqlite');
conn = sqlite(sqlfile);
data = fetch(conn, 'SELECT * FROM Data');

% Process variables
Age = double(data.Age)';
Year = double(data.Year)';
AxilNodes = double(data.AxilNodes)';
y = double(data.SurvivalStatus)';
Z1 = Age - 52;
Z2 = Year - 63;

% Regressor matrix (with dimension 6x306)
X = [Z1; Z1.^2; Z1.^3; Z2; Z1.*Z2; log(1+AxilNodes)];

% Remove observation 8 (see Landwehr, Pregibon and Shoemaker (1984), page 86)
X(:, 8) = [];
y(8) = [];

% Estimation
bStart = 1.5; thetaStart = [0.03; 0.004; -0.0003; 0; 0; -0.5];
[bhat, thetahat, ~, AsymptoticCov] = EstimateLogisticRegression(X, y, bStart, thetaStart, true);
gammahat = [bhat; thetahat];
n = length(y);
StdError = sqrt(diag(AsymptoticCov))/sqrt(n);
pValue = 2*normcdf( -abs(gammahat)./StdError );

% Print results
names = {'intercept', 'z_1', 'z_1^2', 'z_1^3', 'z_2', 'z_1xz_2', 'log(1+x_3)'};
fprintf('%-15s %-15s %-15s %-15s\n', 'variable', 'MLE', 'Std. Error', 'P-value')
fprintf('------------------------------------------------------------------\n')
for iter = 1:length(gammahat)
    fprintf('%-15s %-15.4e %-15.4e %-15.4f \n', names{iter}, gammahat(iter), StdError(iter), pValue(iter));
end
fprintf('\n')

% Confusion matrix
EstProb =  1./(1+exp(-(bhat+thetahat'*X)));
CutOff = 0.5;
yhat = 1.0*(EstProb>=CutOff);
Overview= [y; yhat];
TP = sum((y==1).*(yhat==1));        % Correctly classified 5-year survivor
TN = sum((y==0).*(yhat==0));        % Correctly classified non-survivor
FP = sum((y==0).*(yhat==1));        % Erroneously classified as survivor
FN = sum((y==1).*(yhat==0));        % Erroneously classified as non-survivor
assert( (TP+TN+FP+FN)==n, 'Confusion matrix error: entry count does not match the number of observations')
fprintf('\n\n%-6s|%10s %10s\n', '', 'yhat=1', 'yhat=0')
fprintf('-----------------------------------------\n')
fprintf('%-6s|%10d %10d\n', 'y=1', TP, FN)
fprintf('%-6s|%10d %10d\n', 'y=0', FP, TN)
fprintf('Observations: %-d\n', n)
fprintf('Accuracy: %-4.4g%%\n\n', 100*(TP+TN)/n)

% Figure illustrating Age effect
x1list = -25:35;
age1list = thetahat(1)*x1list + thetahat(2)*x1list.^2 + thetahat(3)*x1list.^3;
EstSubset = [2; 3; 5; 6];
bStartRestr = bStart; thetaStartRestr = thetaStart(EstSubset);
XRestr = X(EstSubset, :);
[bhatRestr, thetahatRestr] = EstimateLogisticRegression(XRestr, y, bStartRestr, thetaStartRestr, true);
age2list = thetahatRestr(1)*x1list.^2+thetahatRestr(2)*x1list.^3;
plot(x1list, age1list, 'LineWidth', 2)
hold on
plot(x1list, age2list, 'LineWidth', 2)
hold off
box on
grid on
xticks([-40 -20 0 20 40])
set(gca, 'FontSize', 15)
xlabel('(Age-52)', 'FontSize', 20)
ylabel('Age contribution to $\Lambda$', 'FontSize', 25, 'Interpreter', 'latex')

