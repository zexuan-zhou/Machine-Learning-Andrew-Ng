
function [J grad] = nnCostFunction(nn_params, ...
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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% X_designMatrix = [ones(length(X(:,1)),1) X]; % add ones to the X vector
% X_designMatrix 5000x401
% a_1 = X_designMatrix; % a_1 is the input layer
a_1 = [ones(length(X(:,1)),1) X];
% a_1 5000x401
% Theta1 25x401
z_2 = Theta1 * a_1';  % z_2 is designed matrix for the hidden layer and it's 
% z_2 25x5000
a_2 = sigmoid(z_2); % a_2 is the activation in the hidden layer
% a_2 25x5000
a_2 = [ones(1, length(a_2(1,:))); a_2]; % a_2 26x5000.
% Theta2 10x26
z_3 = Theta2 * a_2; % z_3 10x5000.
a_3 = sigmoid(z_3); % a_3 is the output layer
% a_3 10x5000.

% Transform y into vector form
% if y = 1, then y_vec = [1 0 0 ... 0]
% if y = 2, then y_vec = [0 1 0 ... 0] so on
y_vec = zeros(length(y(:,1)),num_labels); % initialize y_vec to 0
for sample_i = 1 : length(y(:,1)) % 5000 training samples
    for j = 1 : num_labels
        if y(sample_i) == j 
            y_vec(sample_i,j) = 1; % y_vec(sample_i,:) should be 1x10.
        end
    end
end

K = num_labels;
for sample_i = 1 : length(y)
    y_i = y_vec(sample_i,:);
    h_i = a_3(:,sample_i);
   % J = J + (-y_i * log(h_i) - (1 - y_i) * log(1 - h_i));
   for k = 1 : K
       J = J + (-y_i(k) * log(h_i(k)) - (1 - y_i(k)) * log(1 - h_i(k)));
   end
end
J = 1/m * J; % finish calculating cost function J WITHOUT regularization.

% Adding the regularization term
regularization_theta1 = sum(sum(Theta1(:,2:end).^2)); % start calculating regularization term for theta1.
%for j = 1 : length(Theta1(:,1))
    %regularization_theta1 = regularization_theta1 + Theta1(j, 2:end) * Theta1(j, 2:end)';
%end
regularization_theta2 = sum(sum(Theta2(:,2:end).^2)); % start calculating regularization term for theta2.
%for j = 1 : length(Theta2(1,2:end))
    %regularization_theta2 = regularization_theta2 + Theta2(2:end, j)' * Theta2(2:end, j);
%end
J = J + lambda/(2 * m) * (regularization_theta1 + regularization_theta2);

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

Delta_1 = 0; Delta_2 = 0; % Initialize capital delta.
for t = 1 : m % iterate all training data.
    
    % Step 1: set the input layer's values.
    a1 = X(t,:); % set a(1) to the t-th training example x(t).
    a1 = [1, a1]; % add the bias parameter. Note that a_1 is a row vector.
    % a_1 is 1x401. The first col in a_1 is the bias parameter.
    z2 = Theta1 * a1'; % z_2 is 25x1.
    a2 = sigmoid(z2);
    a2 = [1; a2]; % add the bias parameter. Note that a_2 is a col vector.
    % a_2 is 26x1. The first row in a_2 is the bias parameter.
    z3 = Theta2 * a2; % z_3 is 10x1.
    a3 = sigmoid(z3); % a_3 is 10x1.
    % Step 1 finished.
    
    % Step 2: calculate error in output layer (i.e. layer 3).
    % Note that the y_vec(sample_i,:) defined above is 1x10.
    % Note that the a_3 is 10x1.
    delta_3 = a3 - y_vec(t,:)'; % delta_3 is 10x1.
    % Step 2 finished.
    
    % Step 3: calculate error in the hidden layer (i.e. layer 2)
    % Theta2 is 10x26, delta_3 is 10x1, z_2 is 25x1.
    delta_2 = Theta2' * delta_3; % 26x1.
    delta_2 = delta_2(2:end); % remove the bias parameter. 25x1.
    delta_2 = delta_2 .* sigmoidGradient(z2); % calculating the error in layer 2.
    % Step 3 finished.
    
    % Step 4: accumulate the gradient.
    % Calculating the gradient.
    % By formula we have: (let l denote the l-th layer)
    % Delta_l = Delta_l + delta_(l+1)*(a_l)'.
    % delta_2 is 25x1, a_1 is 1x401.
    % Don't need the bias parameter in a1
    Delta_1 = Delta_1 + delta_2 * a1; % 25x401.
    % delta_3 is 10x1, a_2 is 26x1.
    % Don't need the bias parameter in a1
    Delta_2 = Delta_2 + delta_3 * a2'; % 10x26.
    % Step 4 finished.
    % Step 1 to 4 finished (these steps must be done within the for-loop).
end

    % Step 5 : accumulate the gradient (w/ regularization)
    % Obtain the gradient for the neural network cost function by dividing
    % the accumulated gradients by m. Also we need to add the regularization term:
    % Delta_1 25x401. Theta1 25x401.
    % Delta_2 10x26. Theta2 10x26.
    Theta1_grad = 1/m * Delta_1 + lambda/m * [zeros(length(Theta1(:,1)),1) Theta1(:,2:end)]; % don't need to regularize the bias term
    Theta2_grad = 1/m * Delta_2 + lambda/m * [zeros(length(Theta2(:,1)),1) Theta2(:,2:end)]; % don't need to regularize the bias term
    grad = [Theta1_grad(:);Theta2_grad(:)]; 
    % Step 5 finished.
    
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

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
