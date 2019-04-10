%% This test of linear regression works as following: 
% randomly generated theta values, random design matrixes, random result values (as linear function from theta with noise)
% then design matrix and resulting vector are stored and gradient descent is used
%% Clear and Close Figures
clear ; close all; clc

fprintf('Generating data ...\n');

%% Load Data
m=100;
n=10;
X = 1*rand(m, n);
X = [ones(m, 1) X];
theta=10*rand(n+1, 1);
y=zeros(m, 1);
for i=1:m
  y(i)=X(i,:)*theta + 1*randn();
endfor
fprintf('Generated theta:\n');
fprintf('%f\n', theta);
fprintf('Cost of exact values:\n');
fprintf('%f\n', computeCostMulti(X, y, theta));



% Print out some data points
%fprintf('First 5 examples from the dataset: \n');
%fprintf(' x = [%f %f], y = %f \n', [X(1:5,:) y(1:5,:)]');
%fprintf('Program paused. Press enter to continue.\n');
%pause;


% Scale features and set them to zero mean
%fprintf('Normalizing Features ...\n');
%[X mu sigma] = featureNormalize(X);
%fprintf('mu is:%f\n', mu);
%fprintf('sigma is:%f\n', sigma);


fprintf('Running gradient descent ...\n');
alpha = 0.3;
num_iters = 1000;
theta = zeros(n+1, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
fprintf('Cost of gradient descent values:\n');
fprintf('%f\n', computeCostMulti(X, y, theta));


fprintf('Solving with normal equations...\n');

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
fprintf('Cost of normal equations values:\n');
fprintf('%f\n', computeCostMulti(X, y, theta));

