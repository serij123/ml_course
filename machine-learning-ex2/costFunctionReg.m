function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

thetaWoFirst = [0;theta(2:size(theta))];%theta with zeroed first element - we're not going to regilarize this element

% You need to return the following variables correctly 
J = -(y'*log(sigmoid(X*theta)) + (1-y)'*log(1-sigmoid(X*theta)))/m + lambda*thetaWoFirst'*thetaWoFirst/(2*m);
grad = X'*(sigmoid(X*theta)-y)/m + lambda*thetaWoFirst/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
