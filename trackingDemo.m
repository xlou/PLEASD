%% Bundle method for risk minimization
load('data/data-tracking-training-test.mat');

clear settings functors;

% Settings
settings.lambda = 1;        % L2 regularization strength
settings.augment = 1;       % specify augment problem
settings.search_space = 0;  % specify search space
settings.epsilon = 1e-6;    % convergence condition (approximation gap)
settings.max_iter = 250;    % maximum number of iterations
settings.verbose = 1;       % verbose output

% User provided functors
functors.loss = @trackingLoss;
functors.joint_feature = @trackingJointFeature;
functors.predictor = @trackingPredictor;

% Initialize: get the first prediction
w = zeros(sum(colsInCell(Dtraining{1}.x)), 1);
for n = 1:length(Dtraining)
    d = Dtraining{n};
    d.y_dot = d.y_full;
    d.y_hat = trackingPredictor(d, w, settings.augment, settings.search_space);
    Dtraining(n) = {d};
end

ticID = tic;
[w, Dtraining, A, B, R, W, cccp_meta] = bmrm(w, Dtraining, functors, settings);
println('\nTotal training time: %g', toc(ticID));
println('Training task loss: %g', trackingGetFullLoss(Dtraining, w, functors.predictor, functors.loss));
println('Test task loss: %g', trackingGetFullLoss(Dtest, w, functors.predictor, functors.loss));

%% Structured learning from partial annotations
load('data/data-tracking-training-test.mat');

clear settings functors;

% Settings
settings.reward_augment = 0;        % specify augment problem for reward
settings.reward_search_space = 1;   % specify search space for reward
settings.penalty_augment = 1;       % specify augment problem for penalty
settings.penalty_search_space = 0;  % specify search space for penalty
settings.lambda = 1;                % L2 regularization strength
settings.eta = 1e-3;                % covergence condition (change of w)
settings.epsilon = 1e-3;            % covergence condition of bmrm
settings.max_iter = 250;            % maximum number of iterations
settings.verbose = 1;               % verbose output
settings.reuse_cuts = 1;            % reuse cuts
settings.adaptive_epsilons = ...    % adaptive precision
    max((1/2).^(0:settings.max_iter-1), settings.epsilon);

% User provided functors
functors.loss = @trackingLoss;
functors.joint_feature = @trackingJointFeature;
functors.predictor = @trackingPredictor;

% Initialize: get the first prediction
w = zeros(sum(colsInCell(Dtraining{1}.x)), 1);
for n = 1:length(Dtraining)
    d = Dtraining{n};
    d.y_dot = d.y_partial;
    d.y_hat = trackingPredictor(d, w, settings.penalty_augment, settings.penalty_search_space);
    Dtraining(n) = {d};
end

ticID = tic;
[w, Dtraining, A, B, R, W, cccp_meta] = slpa_bmrm(w, Dtraining, functors, settings);
println('\nTotal training time: %g', toc(ticID));
println('Training loss: %g', trackingGetFullLoss(Dtraining, w, functors.predictor, functors.loss));
println('Test loss: %g', trackingGetFullLoss(Dtest, w, functors.predictor, functors.loss));

%% Structured perceptron

load('data/data-tracking-training-test.mat');

clear settings functors;

% Settings
settings.augment = 1;       % specify augment problem
settings.search_space = 0;  % specify search space
settings.eta = 1e-5;        % convergence condition (change of w)
settings.rho = 1e-5;        % convergence condition (training error)
settings.max_iter = 250;    % maximum number of iterations
settings.verbose = 1;       % verbose output

% User provided functors
functors.loss = @trackingLoss;
functors.joint_feature = @trackingJointFeature;
functors.predictor = @trackingPredictor;

% Initialize: get the first prediction
w = zeros(sum(colsInCell(Dtraining{1}.x)), 1);
for n = 1:length(Dtraining)
    d = Dtraining{n};
    d.y_dot = d.y_full;
    d.y_hat = trackingPredictor(d, w, settings.augment, settings.search_space);
    Dtraining(n) = {d};
end

ticID = tic;
[w, Dtraining, perceptron_meta] = perceptron(w, Dtraining, functors, settings);
println('\nTotal training time: %g', toc(ticID));
println('Training loss: %g', trackingGetFullLoss(Dtraining, w, functors.predictor, functors.loss));
println('Test loss: %g', trackingGetFullLoss(Dtest, w, functors.predictor, functors.loss));


%% Structured perceptron for partial annotations

load('data/data-tracking-training-test.mat');

clear settings functors;

% Settings
settings.reward_augment = 0;        % specify augment problem for reward
settings.reward_search_space = 1;   % specify search space for reward
settings.penalty_augment = 1;       % specify augment problem for penalty
settings.penalty_search_space = 0;  % specify search space for penalty
settings.eta = 1e-5;                % covergence condition (change of w)
settings.rho = 1e-5;                % covergence condition (training error)
settings.max_iter = 250;            % maximum number of iterations
settings.verbose = 1;               % verbose output

% User provided functors
functors.loss = @trackingLoss;
functors.joint_feature = @trackingJointFeature;
functors.predictor = @trackingPredictor;

% Initialize: get the first prediction
w = zeros(sum(colsInCell(Dtraining{1}.x)), 1);
for n = 1:length(Dtraining)
    d = Dtraining{n};
    d.y_dot = d.y_partial;
    d.y_hat = trackingPredictor(d, w, settings.penalty_augment, settings.penalty_search_space);
    Dtraining(n) = {d};
end

ticID = tic;
[w, Dtraining, perceptron_meta] = slpa_perceptron(w, Dtraining, functors, settings);
println('\nTotal training time: %g', toc(ticID));
println('Training loss: %g', trackingGetFullLoss(Dtraining, w, functors.predictor, functors.loss));
println('Test loss: %g', trackingGetFullLoss(Dtest, w, functors.predictor, functors.loss));
