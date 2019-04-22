function [J grad] = nnCostFunctionMatr(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1), X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));


a1=X';
z2=Theta1 * a1;
a2=[ones(1,size(z2,2));sigmoid(z2)];
z3=Theta2 * a2;
a3=sigmoid(z3)';%a3 is the output of neural network
J=sum(sum(y.*log(a3)+(1-y).*log(1-a3)));
delta3=a3'-y';
Delta2=delta3*a2';
delta2=(Theta2'*delta3) .* (a2 .* (1-a2));
delta2=[zeros(1,size(delta2, 2)); delta2(2:end,:)];
dd2=delta2*a1';
Delta1 = dd2(2:end,:);

%%parfor i=1:m
%  %forward propagation part of algorithm
%  J_i = 0;
%  %y_i is desired version of y (for ex if y[i] is 3, y_i should be [0 0 1 0 0...0]:
%  y_i = zeros(1, num_labels);
%  y_i(y(i)) = 1;
%  a1 = X(i,:)';
%  z2 = Theta1 * a1;
%  a2 = [ones(1,1); sigmoid(z2)];
%  z3 = Theta2 * a2;
%  a3 = sigmoid(z3);%a3 is the output of neural network
%  J_i = J_i + y_i*log(a3) + (1-y_i)*log(1-a3);
%  J = J + J_i;

%  %backpropagation part of algorithm  
%  delta3=a3-y_i';
%  Delta2 = Delta2 + delta3*a2';
%  delta2 =(Theta2'* delta3) .* (a2 .* (1-a2));
%  delta2 = [0;delta2(2:end)];
  
%  dd2=delta2*a1';
%  Delta1 = Delta1 + dd2(2:end,:);
%%endparfor


J = -J/m;
J_reg = 0;
Theta1WoFirst = [zeros(size(Theta1,1),1), Theta1(:,2:size(Theta1,2))];
Theta2WoFirst = [zeros(size(Theta2,1),1), Theta2(:,2:size(Theta2,2))];
J_reg = lambda * (sum(sum(Theta1WoFirst.*Theta1WoFirst)) + sum(sum(Theta2WoFirst.*Theta2WoFirst))) / (2*m);
J = J + J_reg;


Theta1_grad=(Delta1 + lambda * Theta1WoFirst)/m;
Theta2_grad=(Delta2 + lambda * Theta2WoFirst)/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================



end
