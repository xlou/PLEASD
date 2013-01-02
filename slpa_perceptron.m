function [w, D, perceptron_meta] = slpa_perceptron(w, D, functors, settings)
% [w, D, perceptron_meta] = slpa_perceptron(w, D, functors, settings) is
% the entry function for structured perceptron learning from partial 
% annotations.
% 
% Input:
%       w:          initial parameter
%       D:          input data samples (a cell object)
%       functors:   user provided functors
%       settings:   additional settings (see User Guide) for more details
% 
% Output:
%       w:          learned parameter
%       D:          updated data samples (a cell object)
%       perceptron_meta:    meta information in the learning process
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

% additional settings
penalty_augment = getField(settings, 'penalty_augment', 1);
penalty_search_space = getField(settings, 'penalty_search_space', 0);
reward_augment = getField(settings, 'reward_augment', 0);
reward_search_space = getField(settings, 'reward_search_space', 1);
eta = getField(settings, 'eta_hat', 1e-5);
rho = getField(settings, 'rho_hat', 1e-5);
max_iter = getField(settings, 'max_iter', 250);
verbose = getField(settings, 'verbose', 0);

eta_hat = 2*eta;
rho_hat = 2*rho;

% timing it
tSt = cputime;

iter = 1;
while eta_hat > eta && rho_hat > rho && iter < max_iter,
    println(verbose, '********Perceptron iter %04d********', iter);

    % last w: w_
    w_ = w;
    
    % loop through data
    for n = 1:length(D)
        d = D{n};

        % update prediction for the reward term
        d.y_tilde = functors.predictor(d, w, reward_augment, reward_search_space);

        % update prediction for the penalty term
        d.y_hat = functors.predictor(d, w, penalty_augment, penalty_search_space);

        % update model
        w = w + functors.joint_feature(d.x, d.y_tilde) - functors.joint_feature(d.x, d.y_hat);

        % save the result
        D(n) = {d};
    end

    % update eta_hat and rho_hat
    eta_hat = norm(w - w_) ./ norm(w);
    rho_hat = perceptronGetLoss(D, functors.loss);
    println(verbose, 'eta_hat: %g; rho_hat: %g', eta_hat, rho_hat);

    % save meta info
    perceptron_meta.runtime(iter) = cputime - tSt;
    perceptron_meta.w(iter) = {w};
    perceptron_meta.eta_hat(iter) = eta_hat;
    perceptron_meta.J(iter) = rho_hat;
    
    % count iterations
    iter =  iter + 1;
end
