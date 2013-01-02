function [w, D, A, B, R, W, bmrm_meta] = bmrm(w, D, functors, settings, A, B, R, W)
% [w, D, A, B, R, W, bmrm_meta] = bmrm(w, D, functors, settings, A, B, R, W)
% is the entry function for structured learning using bundle method for 
% risk minimization.
% 
% Input:
%       w:          initial parameter
%       D:          input data samples (a cell object)
%       functors:   user provided functors
%       settings:   additional settings (see User Guide) for more details
%       A, B:       initial gradients and offsets as the linear lower bound
%       R, W:       initial risk (loss) and parameters w.r.t A and B
% 
% Output:
%       w:          learned parameter
%       D:          updated data samples (a cell object)
%       A, B:       updated gradients and offsets as the linear lower bound
%       R, W:       updated risk (loss) and parameters w.r.t A and B
%       cccp_meta:  meta information in the learning process
% 
% Hints:
% [w, D, A, B, R, W, bmrm_meta] = bmrm(w, D, functors, settings) assumes no
% initial lower bounds, i.e. setting A, B, R and W to empty.
%
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

% initialization
if nargin <= 4
    A = []; B = [];
    R = []; W = [];
end

% additional settings
lambda = getField(settings, 'lambda', 1);
v_extra = getField(settings, 'v_extra', zeros(size(w)));
augment = getField(settings, 'augment', 1);
search_space = getField(settings, 'search_space', 0);
epsilon = getField(settings, 'epsilon', 1e-5);
max_iter = getField(settings, 'max_iter', 250);
verbose = getField(settings, 'verbose', 0);
inside_cccp = getField(settings, 'inside_cccp', 0);

% compute approximation gap
epsilon_hat = 2*epsilon;

% timing it
ticID = tic;

iter = 1;
while epsilon_hat > epsilon && iter < max_iter,
    println(verbose, '****BMRM iter %04d****', iter);

    % compute subgradients
    [a, b] = bmrmGetGradient(D, w, functors.joint_feature, functors.loss, inside_cccp);
    A = [A, {a}];
    B = [B, {b}];

    % update model parameter
    w = bmrmUpdateModelL2(A, B, lambda, v_extra, inside_cccp);
    W = [W, {w}];

    % update predictions using the new w
    for n = 1:length(D)
        d = D{n};
        d.y_hat = functors.predictor(d, w, augment, search_space);
        D(n) = {d};
    end

    % compute main objective
    r = bmrmGetLoss(D, w, functors.joint_feature, functors.loss, inside_cccp);
    R = [R, {r}];

    % compute approximation gap
    J = cell2mat(R);
    for i = 1:length(J)
        J(i) = J(i) + lambda * norm(W{i}).^2/2 + dot(W{i}, v_extra);
    end
    J_hat = max(cell2mat(A)' * w + cell2mat(B)') + ...
        lambda * norm(w).^2/2 + dot(w, v_extra);
    epsilon_hat = min(J) - J_hat;
    println(verbose, '\tepsilon = %g', epsilon_hat);
    
    % save meta info
    bmrm_meta.runtime(iter) = toc(ticID);
    bmrm_meta.epsilon_hat(iter) = epsilon_hat;
    bmrm_meta.risk(iter) = r;
    bmrm_meta.J = J;
    bmrm_meta.w(iter) = {w};

    % count iterations
    iter =  iter + 1;
end
