function res = trackingGetFullLoss(D, w, functorPredictor, functorLoss)
% res = trackingGetFullLoss(X, w, functorPredictor, functorLoss) predicts
% the optimal output using given parameter w and compute the average loss
% w.r.t. the full annotations.
% 
% Input:
%       D:          input data samples (a cell object)
%       w:          parameter vector
%       functorPredictor:	predictor functor
%       functorLoss:        loss functor
% 
% Output:
%       res:        average loss
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

N = length(D);

% predict the optimal output and compute the average loss
res = 0;
for n = 1:N
    d = D{n};
    d.y_hat = functorPredictor(d, w);
    res = res + functorLoss(d.y_full, d.y_hat);
end
res = res / N;
