function res = perceptronGetLoss(D, functorLoss)
% res = perceptronGetLoss(D, functorLoss) returns the average loss for
% perceptron learning using exising predictions in D.
% 
% Input:
%       D:          input data samples (a cell object)
%       functorLoss:	loss functor
% 
% Output:
%       res:        average loss
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

N = length(D);

% compute the gradient and offset
res = 0;
for n = 1:N
    d = D{n};

    res = res + functorLoss(d.y_dot, d.y_hat);
end

res = res / N;
