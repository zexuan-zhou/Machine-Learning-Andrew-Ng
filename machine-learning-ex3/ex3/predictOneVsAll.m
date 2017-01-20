function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% M returns the max value of each row, p returns the corresponding indices
% of the number of labels.
% M means the largest probability of the i-th test element being predicted
% as the p-th element in the labels
% In this exercise, p = [10 1 2 3 4 5 6 7 8 9], 
% correponding to numbers from 0 to 9.
% For example, if the first element in M is 0.95 and the first element in p is
% 10, this means that the firs element in the test set is predicted to be 0 
% with a probablity of 0.95, and this probability is larger than any other 
% probability that this test element has to be predicted as one of 1 to 9.
[M, p] = max(sigmoid(X * all_theta'), [], 2);

% =========================================================================


end
