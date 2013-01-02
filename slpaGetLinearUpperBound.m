function a = slpaGetLinearUpperBound(D, functorJointFeature)
% a = slpaGetLinearUpperBound(D, functorJointFeature) computes the gradient
% using d.y_tilde which upper bounds the concave part of the objective
% function. Used in structured learning from partial annotations.
% 
% Input:
%       D:          input data samples (a cell object)
%       functorJointFeature:    joint feature functor
% 
% Output:
%       a:          gradient
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

N = length(D);

% compute the gradient
a = 0; 
for n = 1:N
    d = D{n};
    
    Psi = functorJointFeature(d.x, d.y_tilde);
    a = a + Psi;

    D(n) = {d};
end

a = - a / N; 
