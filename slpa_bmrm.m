function [w, D, A, B, R, W, cccp_meta] = slpa_bmrm(w, D, functors, settings, A, B, R, W)
% [w, D, A, B, R, W, cccp_meta] = slpa_bmrm(w, D, functors, settings, A, B, R, W)
% is the entry function for structured learning from partial annotations.
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
% [w, D, A, B, R, W, cccp_meta] = slpa_bmrm(w, D, functors, settings) assumes no
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
reward_augment = getField(settings, 'reward_augment', 0);
reward_search_space = getField(settings, 'reward_search_space', 1);
penalty_augment = getField(settings, 'penalty_augment', 1);
penalty_search_space = getField(settings, 'penalty_search_space', 0);
lambda = getField(settings, 'lambda', 1);
eta = getField(settings, 'eta', 1e-5);
epsilon = getField(settings, 'epsilon', 1e-5);
max_iter = getField(settings, 'max_iter', 250);
verbose = getField(settings, 'verbose', 0);
reuse_cuts = getField(settings, 'reuse_cuts', 1e-5);
adaptive_epsilons = getField(settings, 'adaptive_epsilons', []);

% setting for the inner bmrm run
bmrm_settings.lambda = lambda;
bmrm_settings.augment = penalty_augment;
bmrm_settings.search_space = penalty_search_space;
bmrm_settings.epsilon = epsilon;
bmrm_settings.max_iter = max_iter;
bmrm_settings.inside_cccp = 1;
bmrm_settings.verbose = verbose;

eta_hat = 2*eta;

% timing it
ticID = tic;

iter = 1;
while eta_hat > eta && iter < max_iter,
    println(verbose, '********CCCP iter %04d********', iter);

    % last w: w_
    w_ = w;

    % update the predicion for the reward term
    for n = 1:length(D)
        d = D{n};
        d.y_tilde = functors.predictor(d, w, reward_augment, reward_search_space);
        D(n) = {d};
    end

    % get the upperbound of the concave function
    v_extra = slpaGetLinearUpperBound(D, functors.joint_feature);
    bmrm_settings.v_extra = v_extra;

    % set the epsilon for bmrm
    if ~isempty(adaptive_epsilons)
        bmrm_settings.epsilon = adaptive_epsilons(iter);
    end
    println(verbose, 'bmrm epsilon = %g', bmrm_settings.epsilon);
    
    % enter bmrm
    if reuse_cuts
        [w, D, A, B, R, W, bmrm_meta] = bmrm(w, D, functors, bmrm_settings, A, B, R, W);
    else
        [w, D_, A_, B_, R_, W_, bmrm_meta] = bmrm(w, D, functors, bmrm_settings, A, B, R, W);
    end

    % update eta_hat
    eta_hat = norm(w - w_);
    println(verbose, 'eta_hat: %g', eta_hat);

    % save meta info
    cccp_meta.runtime(iter) = toc(ticID);
    cccp_meta.cuts(iter) = length(A);
    cccp_meta.v_extra(iter) = {v_extra};
    cccp_meta.w(iter) = {w};
    cccp_meta.eta_hat(iter) = eta_hat;
    cccp_meta.bmrm_meta(iter) = {bmrm_meta};
    cccp_meta.J(iter) = min(bmrm_meta.J);
    
    % count iterations
    iter =  iter + 1;
end
