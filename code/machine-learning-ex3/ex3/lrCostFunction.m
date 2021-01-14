function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



% calculate cost function
h = sigmoid(X*theta);

%X =
%   1.00000   0.10000   0.60000   1.10000
%   1.00000   0.20000   0.70000   1.20000
%   1.00000   0.30000   0.80000   1.30000
%   1.00000   0.40000   0.90000   1.40000
%   1.00000   0.50000   1.00000   1.50000

%theta =
%  -2
%  -1
%   1
%   2

%h =
%   0.66819
%   0.71095
%   0.75026
%   0.78583
%   0.81757

% excluded the first theta value
theta1 = [0 ; theta(2:end, :)];
%theta1 =
%   0
%  -1
%   1
%   2

%y =
%   1
%   0
%   1
%   0
%   1

%lambda = 3

%J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + lambda*(theta1'*theta1)/(2*m);
J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + lambda*(sum(theta1.^2))/(2*m);

%J =  2.5348

%X' =
%   1.00000   1.00000   1.00000   1.00000   1.00000
%   0.10000   0.20000   0.30000   0.40000   0.50000
%   0.60000   0.70000   0.80000   0.90000   1.00000
%   1.10000   1.20000   1.30000   1.40000   1.50000

% calculate grads
grad = (X'*(h - y) + lambda*theta1)/m;

%grad =
%   0.14656
%  -0.54856
%   0.72472
%   1.39800

% =============================================================

grad = grad(:);

end
