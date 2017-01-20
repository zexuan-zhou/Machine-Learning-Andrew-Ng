function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% compute the cost function J
for i = 1 : m
    J = J - y(i) * log(sigmoid(X(i,:) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i,:) * theta));
end
J = J / m;
% compute the regularization part
regularized_theta_sum = 0; % initialize the regularization part
for i = 2 : size(theta)    % i starts from 2 because don't need to regularize theta_0
    regularized_theta_sum = regularized_theta_sum + theta(i)^2;
end
regularized_theta_sum = 0.5 * lambda / m * regularized_theta_sum;
J = J + regularized_theta_sum; % add the regularization part to the cost

% compute partial derivative for each theta
for i = 1 : m
    for j = 1 : size(theta)
    grad(j) = grad(j) + (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
    end
end
for i = 1 : size(theta)
    grad(i) = grad(i) / m;
end
for i = 2 : size(theta) % i starts from 2 because don't need to regularize theta_0
    grad(i) = grad(i) + lambda / m * theta(i); % add the regularization part to the gradient
end
% =============================================================

