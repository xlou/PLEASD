function res = trackingLoss(y_dot, y_hat)
% res = trackingLoss(y_dot, y_hat) returns a scalar that represents the 
% difference between the ground truth y_dot and some output y_hat
% 
% Input:
%       y_dot:      ground truth output
%       y_hat:      predicted output
% 
% Output:
%       res:        a scalar
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

y_dot_vec = cell2mat(y_dot);
y_hat_vec = cell2mat(y_hat);

% compute the difference and normalize it
res = sum(double(y_dot_vec ~= y_hat_vec) .* double(y_dot_vec == 1));
res = res ./ sum(y_dot_vec == 1);
