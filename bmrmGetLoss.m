function res = bmrmGetLoss(D, w, functorJointFeature, functorLoss, inside_cccp)
% res = bmrmGetLoss(D, functorLoss) returns the average surrogate loss for
% bmrm learning using exising predictions in D.
% 
% Input:
%       D:          input data samples (a cell object)
%       w:          learned parameter
%       functorJointFeature:    joint feature vector
%       functorLoss:	loss functor
%       inside_cccp:    indicating whether bmrm is inside a cccp procedure
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

    loss = functorLoss(d.y_dot, d.y_hat);
    
    % note that the scoring is different between a pure bmrm and bmrm iside
    % a cccp procedure
    Psi_hat = functorJointFeature(d.x, d.y_hat);
    if ~inside_cccp
        Psi_dot = functorJointFeature(d.x, d.y_dot);
        Psi = Psi_hat - Psi_dot;
    else
        Psi = Psi_hat;
    end

    res = res + loss + dot(w, Psi);
end

res = res / N;
